# frozen_string_literal: true

require "fileutils"
require "net/http"
require "uri"
require "json"

module OnnxRuby
  module Hub
    DEFAULT_CACHE_DIR = File.join(Dir.home, ".cache", "onnx_ruby", "models")

    # Download a model from Hugging Face Hub
    # @param repo_id [String] e.g. "sentence-transformers/all-MiniLM-L6-v2"
    # @param filename [String] ONNX file to download (default: "model.onnx")
    # @param cache_dir [String] local cache directory
    # @return [String] path to the downloaded model file
    def self.download(repo_id, filename: "model.onnx", cache_dir: DEFAULT_CACHE_DIR, revision: "main")
      model_dir = File.join(cache_dir, repo_id.tr("/", "--"), revision)
      model_path = File.join(model_dir, filename)

      return model_path if File.exist?(model_path)

      FileUtils.mkdir_p(model_dir)

      url = "https://huggingface.co/#{repo_id}/resolve/#{revision}/#{filename}"
      download_file(url, model_path)

      model_path
    end

    # List cached models
    # @param cache_dir [String] cache directory to search
    # @return [Array<String>] list of cached model paths
    def self.cached_models(cache_dir: DEFAULT_CACHE_DIR)
      return [] unless Dir.exist?(cache_dir)

      Dir.glob(File.join(cache_dir, "**", "*.onnx"))
    end

    # Clear the model cache
    # @param cache_dir [String] cache directory to clear
    def self.clear_cache(cache_dir: DEFAULT_CACHE_DIR)
      FileUtils.rm_rf(cache_dir) if Dir.exist?(cache_dir)
    end

    class << self
      private

      def download_file(url, dest)
        uri = URI(url)
        max_redirects = 5

        max_redirects.times do
          response = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
            http.request(Net::HTTP::Get.new(uri))
          end

          case response
          when Net::HTTPSuccess
            File.binwrite(dest, response.body)
            return
          when Net::HTTPRedirection
            uri = URI(response["location"])
          else
            raise ModelError, "failed to download #{url}: #{response.code} #{response.message}"
          end
        end

        raise ModelError, "too many redirects downloading #{url}"
      end
    end
  end
end
