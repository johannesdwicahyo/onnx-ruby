# frozen_string_literal: true

require "test_helper"

class TestTensor < Minitest::Test
  def test_create_float_tensor
    tensor = OnnxRuby::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2])
    assert_equal [2, 2], tensor.shape
    assert_equal :float, tensor.dtype
    assert_equal [[1.0, 2.0], [3.0, 4.0]], tensor.to_a
  end

  def test_create_int64_tensor
    tensor = OnnxRuby::Tensor.new([1, 2, 3, 4], shape: [4])
    assert_equal [4], tensor.shape
    assert_equal :int64, tensor.dtype
    assert_equal [1, 2, 3, 4], tensor.to_a
  end

  def test_infer_shape_from_nested_array
    tensor = OnnxRuby::Tensor.new([[1.0, 2.0], [3.0, 4.0]])
    assert_equal [2, 2], tensor.shape
  end

  def test_float_class_method
    tensor = OnnxRuby::Tensor.float([0.1, 0.2, 0.3], shape: [1, 3])
    assert_equal :float, tensor.dtype
    assert_equal [1, 3], tensor.shape
  end

  def test_int64_class_method
    tensor = OnnxRuby::Tensor.int64([1, 2, 3], shape: [3])
    assert_equal :int64, tensor.dtype
  end

  def test_shape_mismatch_raises
    assert_raises(OnnxRuby::TensorError) do
      OnnxRuby::Tensor.new([1, 2, 3], shape: [2, 2])
    end
  end

  def test_unsupported_dtype_raises
    assert_raises(OnnxRuby::TensorError) do
      OnnxRuby::Tensor.new([1, 2], dtype: :complex128)
    end
  end

  def test_flat_data
    tensor = OnnxRuby::Tensor.new([[1.0, 2.0], [3.0, 4.0]])
    assert_equal [1.0, 2.0, 3.0, 4.0], tensor.flat_data
  end

  def test_3d_tensor
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor = OnnxRuby::Tensor.new(data)
    assert_equal [2, 2, 2], tensor.shape
    assert_equal data, tensor.to_a
  end
end
