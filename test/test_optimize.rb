# frozen_string_literal: true

require "test_helper"
require "tmpdir"

class TestOptimize < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
  end

  def test_optimize_model
    Dir.mktmpdir do |dir|
      output_path = File.join(dir, "optimized.onnx")
      result = OnnxRuby.optimize(simple_model_path, output_path)
      assert result
      assert File.exist?(output_path)
      assert File.size(output_path) > 0
    end
  end

  def test_optimize_with_basic_level
    Dir.mktmpdir do |dir|
      output_path = File.join(dir, "optimized_basic.onnx")
      OnnxRuby.optimize(simple_model_path, output_path, level: :basic)
      assert File.exist?(output_path)
    end
  end

  def test_optimized_model_works
    Dir.mktmpdir do |dir|
      output_path = File.join(dir, "optimized.onnx")
      OnnxRuby.optimize(simple_model_path, output_path)

      session = OnnxRuby::Session.new(output_path)
      result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
      assert result.key?("output")
      assert_equal 3, result["output"][0].length
    end
  end

  def test_optimize_nonexistent_model
    assert_raises(OnnxRuby::ModelError) do
      OnnxRuby.optimize("/nonexistent/model.onnx", "/tmp/out.onnx")
    end
  end
end
