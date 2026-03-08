# frozen_string_literal: true

require "test_helper"

class TestProviders < Minitest::Test
  include TestHelper

  def test_available_providers
    providers = OnnxRuby.available_providers
    assert_kind_of Array, providers
    assert_includes providers, "CPUExecutionProvider"
  end

  def test_cpu_provider_explicit
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
    session = OnnxRuby::Session.new(simple_model_path, providers: [:cpu])
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_coreml_provider
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
    skip "CoreML not available" unless RUBY_PLATFORM =~ /darwin/

    providers = OnnxRuby.available_providers
    skip "CoreML provider not available" unless providers.any? { |p| p.include?("CoreML") }

    session = OnnxRuby::Session.new(simple_model_path, providers: [:coreml])
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_unknown_provider_raises
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
    assert_raises(OnnxRuby::Error) do
      OnnxRuby::Session.new(simple_model_path, providers: [:nonexistent])
    end
  end
end
