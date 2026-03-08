# frozen_string_literal: true

require "test_helper"

class TestSessionPool < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
  end

  def test_basic_run
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 2)
    result = pool.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })

    assert result.key?("output")
  end

  def test_with_session_block
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 2)

    pool.with_session do |session|
      assert_kind_of OnnxRuby::Session, session
      result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
      assert result.key?("output")
    end
  end

  def test_pool_size
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 3)
    pool.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })

    assert_equal 1, pool.size
    assert_equal 1, pool.available
  end

  def test_concurrent_access
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 3)
    results = []
    mutex = Mutex.new

    threads = 6.times.map do
      Thread.new do
        r = pool.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
        mutex.synchronize { results << r }
      end
    end
    threads.each(&:join)

    assert_equal 6, results.length
    results.each { |r| assert r.key?("output") }
    assert pool.size <= 3
  end

  def test_sessions_are_reused
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 1)

    3.times { pool.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] }) }

    assert_equal 1, pool.size
  end

  def test_timeout_raises
    pool = OnnxRuby::SessionPool.new(simple_model_path, size: 1, timeout: 0.1)

    blocked = Queue.new
    done = Queue.new

    # Hold the only session
    t = Thread.new do
      pool.with_session do |_s|
        blocked.push(true)
        done.pop  # wait until test is done
      end
    end

    blocked.pop  # wait for thread to hold session

    assert_raises(OnnxRuby::SessionPool::TimeoutError) do
      pool.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    end

    done.push(true)
    t.join
  end
end
