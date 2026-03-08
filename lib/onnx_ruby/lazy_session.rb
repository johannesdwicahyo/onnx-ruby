# frozen_string_literal: true

module OnnxRuby
  class LazySession
    def initialize(model_path, **opts)
      @model_path = model_path
      @opts = opts
      @session = nil
      @mutex = Mutex.new
    end

    def inputs
      load_session.inputs
    end

    def outputs
      load_session.outputs
    end

    def run(inputs, **kwargs)
      load_session.run(inputs, **kwargs)
    end

    def loaded?
      !@session.nil?
    end

    private

    def load_session
      return @session if @session

      @mutex.synchronize do
        @session ||= Session.new(@model_path, **@opts)
      end
    end
  end
end
