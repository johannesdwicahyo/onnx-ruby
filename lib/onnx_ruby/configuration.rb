# frozen_string_literal: true

module OnnxRuby
  class Configuration
    attr_accessor :models_path, :default_providers, :default_log_level

    def initialize
      @models_path = "app/models/onnx"
      @default_providers = [:cpu]
      @default_log_level = :warning
      @pool_size = 5
      @pool_timeout = 5
    end

    def pool_size
      @pool_size
    end

    def pool_size=(value)
      raise ArgumentError, "pool_size must be a positive Integer" unless value.is_a?(Integer) && value > 0

      @pool_size = value
    end

    def pool_timeout
      @pool_timeout
    end

    def pool_timeout=(value)
      raise ArgumentError, "pool_timeout must be a positive Numeric" unless value.is_a?(Numeric) && value > 0

      @pool_timeout = value
    end
  end
end
