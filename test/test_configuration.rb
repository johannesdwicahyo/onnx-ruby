# frozen_string_literal: true

require "test_helper"

class TestConfiguration < Minitest::Test
  def teardown
    # Reset configuration
    OnnxRuby.instance_variable_set(:@configuration, nil)
  end

  def test_default_models_path
    assert_equal "app/models/onnx", OnnxRuby.configuration.models_path
  end

  def test_default_providers
    assert_equal [:cpu], OnnxRuby.configuration.default_providers
  end

  def test_default_pool_size
    assert_equal 5, OnnxRuby.configuration.pool_size
  end

  def test_configure_block
    OnnxRuby.configure do |c|
      c.models_path = "/custom/path"
      c.default_providers = [:coreml, :cpu]
      c.pool_size = 10
    end

    assert_equal "/custom/path", OnnxRuby.configuration.models_path
    assert_equal [:coreml, :cpu], OnnxRuby.configuration.default_providers
    assert_equal 10, OnnxRuby.configuration.pool_size
  end

  def test_configure_pool_timeout
    OnnxRuby.configure { |c| c.pool_timeout = 10 }
    assert_equal 10, OnnxRuby.configuration.pool_timeout
  end

  def test_configure_log_level
    OnnxRuby.configure { |c| c.default_log_level = :error }
    assert_equal :error, OnnxRuby.configuration.default_log_level
  end
end
