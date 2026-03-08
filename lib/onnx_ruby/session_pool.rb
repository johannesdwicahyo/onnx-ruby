# frozen_string_literal: true

module OnnxRuby
  class SessionPool
    class TimeoutError < Error; end

    def initialize(model_path, size: nil, timeout: nil, **session_opts)
      @model_path = model_path
      @session_opts = session_opts
      @size = size || OnnxRuby.configuration.pool_size
      @timeout = timeout || OnnxRuby.configuration.pool_timeout
      @pool = []
      @mutex = Mutex.new
      @condition = ConditionVariable.new
      @created = 0
    end

    # Check out a session, yield it, then check it back in
    def with_session(&block)
      session = checkout
      begin
        yield session
      ensure
        checkin(session)
      end
    end

    # Run inference using a pooled session
    def run(inputs, **kwargs)
      with_session { |s| s.run(inputs, **kwargs) }
    end

    # Current pool stats
    def size
      @mutex.synchronize { @created }
    end

    def available
      @mutex.synchronize { @pool.size }
    end

    private

    def checkout
      @mutex.synchronize do
        loop do
          # Return an available session
          return @pool.pop unless @pool.empty?

          # Create a new one if under limit
          if @created < @size
            @created += 1
            return create_session
          end

          # Wait for one to be returned
          unless @condition.wait(@mutex, @timeout)
            raise TimeoutError, "timed out waiting for session (pool size: #{@size})"
          end
        end
      end
    end

    def checkin(session)
      @mutex.synchronize do
        @pool.push(session)
        @condition.signal
      end
    end

    def create_session
      Session.new(@model_path, **@session_opts)
    end
  end
end
