# frozen_string_literal: true

require "test_helper"
require "tmpdir"

class TestHub < Minitest::Test
  def test_cached_models_empty_dir
    Dir.mktmpdir do |dir|
      result = OnnxRuby::Hub.cached_models(cache_dir: dir)
      assert_equal [], result
    end
  end

  def test_cached_models_nonexistent_dir
    result = OnnxRuby::Hub.cached_models(cache_dir: "/nonexistent/path")
    assert_equal [], result
  end

  def test_clear_cache
    Dir.mktmpdir do |dir|
      cache_dir = File.join(dir, "cache")
      FileUtils.mkdir_p(cache_dir)
      File.write(File.join(cache_dir, "test.onnx"), "fake")

      OnnxRuby::Hub.clear_cache(cache_dir: cache_dir)
      refute Dir.exist?(cache_dir)
    end
  end

  def test_download_returns_cached_path
    Dir.mktmpdir do |dir|
      # Pre-create a fake cached model
      model_dir = File.join(dir, "test-model", "main")
      FileUtils.mkdir_p(model_dir)
      model_path = File.join(model_dir, "model.onnx")
      File.write(model_path, "fake model data")

      result = OnnxRuby::Hub.download("test/model", cache_dir: dir)
      assert_equal model_path, result
    end
  end

  def test_default_cache_dir
    assert_kind_of String, OnnxRuby::Hub::DEFAULT_CACHE_DIR
    assert OnnxRuby::Hub::DEFAULT_CACHE_DIR.include?(".cache/onnx_ruby")
  end
end
