# frozen_string_literal: true

require_relative "lib/onnx_ruby/version"

Gem::Specification.new do |spec|
  spec.name = "onnx-ruby"
  spec.version = OnnxRuby::VERSION
  spec.authors = ["Johannes Dwi Cahyo"]
  spec.email = ["johannesdwicahyo@gmail.com"]

  spec.summary = "Ruby bindings for ONNX Runtime"
  spec.description = "High-performance ONNX Runtime bindings for Ruby using Rice. " \
                     "Run ONNX models locally for embeddings, classification, NER, and more."
  spec.homepage = "https://github.com/johannesdwicahyo/onnx-ruby"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) ||
        f.start_with?("test/", "spec/", "features/", ".git", ".github", "script/")
    end
  end

  spec.require_paths = ["lib"]
  spec.extensions = ["ext/onnx_ruby/extconf.rb"]

  spec.add_dependency "rice", ">= 4.0"

  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "minitest", "~> 5.0"
end
