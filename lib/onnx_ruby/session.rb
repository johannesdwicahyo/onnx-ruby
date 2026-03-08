# frozen_string_literal: true

module OnnxRuby
  class Session
    VALID_PROVIDERS = %i[cpu coreml cuda tensorrt].freeze

    def initialize(model_path, providers: [:cpu], inter_threads: nil, intra_threads: nil,
                   log_level: :warning, optimization_level: :all, memory_pattern: true,
                   cpu_mem_arena: true, execution_mode: :sequential)
      model_path = File.expand_path(model_path)
      raise ModelError, "model file not found: #{model_path}" unless File.exist?(model_path)

      provider_strs = Array(providers).map do |p|
        p = p.to_sym
        raise Error, "unknown provider: #{p}. Valid: #{VALID_PROVIDERS.join(", ")}" unless VALID_PROVIDERS.include?(p)
        p.to_s
      end

      @session = Ext::SessionWrapper.new(
        model_path,
        log_level_to_int(log_level),
        intra_threads || 0,
        inter_threads || 0,
        optimization_level.to_s,
        memory_pattern,
        cpu_mem_arena,
        execution_mode.to_s,
        provider_strs
      )
    end

    def inputs
      @session.input_info
    end

    def outputs
      @session.output_info
    end

    def run(inputs, output_names: nil)
      input_values = inputs.map do |name, data|
        if data.is_a?(Tensor)
          { name: name, data: data.flat_data, shape: data.shape, dtype: data.dtype.to_s }
        else
          flat = data.flatten
          shape = infer_shape(data)
          dtype = infer_dtype(flat)
          { name: name, data: flat, shape: shape, dtype: dtype }
        end
      end

      @session.run(input_values, output_names || [])
    end

    private

    def log_level_to_int(level)
      case level
      when :verbose then 0
      when :info then 1
      when :warning then 2
      when :error then 3
      when :fatal then 4
      else 2
      end
    end

    def infer_shape(data)
      shape = []
      current = data
      while current.is_a?(Array)
        shape << current.length
        current = current.first
      end
      shape
    end

    def infer_dtype(flat)
      sample = flat.find { |v| !v.nil? }
      case sample
      when Float then "float"
      when Integer then "int64"
      when String then "string"
      when true, false then "bool"
      else "float"
      end
    end
  end
end
