# frozen_string_literal: true

module OnnxRuby
  class Configuration
    attr_accessor :models_path, :default_providers, :default_log_level,
                  :pool_size, :pool_timeout

    def initialize
      @models_path = "app/models/onnx"
      @default_providers = [:cpu]
      @default_log_level = :warning
      @pool_size = 5
      @pool_timeout = 5
    end
  end
end
