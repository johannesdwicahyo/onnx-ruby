# frozen_string_literal: true

require "test_helper"

class TestInference < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found — run: python script/create_test_models.py" unless model_exists?("simple.onnx")
    @session = OnnxRuby::Session.new(simple_model_path)
  end

  def test_basic_inference
    result = @session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
    output = result["output"]
    assert_kind_of Array, output
    assert_equal 1, output.length
    assert_equal 3, output[0].length
  end

  def test_batch_inference
    result = @session.run({
      "input" => [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    })
    output = result["output"]
    assert_equal 2, output.length
    assert_equal 3, output[0].length
  end

  def test_with_tensor_input
    tensor = OnnxRuby::Tensor.float([1.0, 2.0, 3.0, 4.0], shape: [1, 4])
    result = @session.run({ "input" => tensor })
    assert result.key?("output")
  end

  def test_with_output_names_filter
    result = @session.run(
      { "input" => [[1.0, 2.0, 3.0, 4.0]] },
      output_names: ["output"]
    )
    assert_equal ["output"], result.keys
  end
end
