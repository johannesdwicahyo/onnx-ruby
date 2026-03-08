# frozen_string_literal: true

module OnnxRuby
  # ActiveModel-style mixin for embedding generation.
  #
  # Usage:
  #   class Document
  #     include OnnxRuby::Model
  #
  #     onnx_model "embeddings.onnx"
  #     onnx_input ->(doc) { { "input_ids" => doc.token_ids, "attention_mask" => doc.mask } }
  #     onnx_output "embeddings"
  #   end
  #
  #   doc = Document.new
  #   doc.onnx_predict  # => [0.123, -0.456, ...]
  module Model
    def self.included(base)
      base.extend(ClassMethods)
    end

    module ClassMethods
      def onnx_model(path = nil, **opts)
        if path
          @onnx_model_path = path
          @onnx_session_opts = opts
        end
        @onnx_model_path
      end

      def onnx_input(callable = nil, &block)
        @onnx_input_fn = callable || block if callable || block
        @onnx_input_fn
      end

      def onnx_output(name = nil)
        @onnx_output_name = name if name
        @onnx_output_name
      end

      def onnx_session
        @onnx_session ||= begin
          path = resolve_model_path(@onnx_model_path)
          LazySession.new(path, **(@onnx_session_opts || {}))
        end
      end

      private

      def resolve_model_path(path)
        return path if File.absolute_path?(path) && File.exist?(path)

        full = File.join(OnnxRuby.configuration.models_path, path)
        return full if File.exist?(full)

        path
      end
    end

    def onnx_predict(**run_opts)
      input_fn = self.class.onnx_input
      raise Error, "onnx_input not defined on #{self.class}" unless input_fn

      inputs = input_fn.call(self)
      result = self.class.onnx_session.run(inputs, **run_opts)

      output_name = self.class.onnx_output
      output_name ? result[output_name] : result.values.first
    end
  end
end
