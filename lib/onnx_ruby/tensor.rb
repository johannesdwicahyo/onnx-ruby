# frozen_string_literal: true

module OnnxRuby
  class Tensor
    DTYPE_MAP = {
      float32: :float,
      float: :float,
      float64: :double,
      double: :double,
      int32: :int32,
      int: :int32,
      int64: :int64,
      bool: :bool,
      string: :string
    }.freeze

    attr_reader :shape, :dtype

    def initialize(data, shape: nil, dtype: nil)
      @data = data.flatten
      @shape = shape || infer_shape(data)
      @dtype = normalize_dtype(dtype || infer_dtype(@data))

      validate!
    end

    def to_a
      reshape(@data.dup, @shape)
    end

    def flat_data
      @data
    end

    def self.float(data, shape: nil)
      new(data, shape: shape, dtype: :float)
    end

    def self.int64(data, shape: nil)
      new(data, shape: shape, dtype: :int64)
    end

    def self.int32(data, shape: nil)
      new(data, shape: shape, dtype: :int32)
    end

    def self.double(data, shape: nil)
      new(data, shape: shape, dtype: :double)
    end

    private

    def normalize_dtype(dtype)
      DTYPE_MAP.fetch(dtype) { raise TensorError, "unsupported dtype: #{dtype}" }
    end

    def infer_shape(data)
      shape = []
      current = data
      while current.is_a?(Array)
        shape << current.length
        # Check for jagged arrays: all sub-arrays at this level must have the same length
        if current.length > 1 && current.all? { |el| el.is_a?(Array) }
          lengths = current.map(&:length).uniq
          if lengths.size > 1
            raise TensorError,
                  "jagged array detected: sub-arrays have lengths #{lengths.sort.join(', ')} at dimension #{shape.size - 1}"
          end
        end
        current = current.first
      end
      shape
    end

    def infer_dtype(flat)
      sample = flat.find { |v| !v.nil? }
      case sample
      when Float then :float
      when Integer then :int64
      when String then :string
      when true, false then :bool
      else :float
      end
    end

    def validate!
      expected_size = @shape.reduce(1, :*)
      if @data.length != expected_size
        raise TensorError,
              "data size #{@data.length} does not match shape #{@shape} (expected #{expected_size})"
      end
    end

    def reshape(flat, dims)
      return flat if dims.length <= 1
      size = dims[1..].reduce(1, :*)
      flat.each_slice(size).map { |slice| reshape(slice, dims[1..]) }
    end
  end
end
