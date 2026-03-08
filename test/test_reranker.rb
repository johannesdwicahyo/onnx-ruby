# frozen_string_literal: true

require "test_helper"

class TestReranker < Minitest::Test
  include TestHelper

  RERANKER_MODEL = File.join(MODELS_DIR, "reranker.onnx")

  def setup
    skip "reranker.onnx not found" unless File.exist?(RERANKER_MODEL)
  end

  def test_score_with_raw_inputs
    reranker = OnnxRuby::Reranker.new(RERANKER_MODEL)

    scores = reranker.score(
      input_ids: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
      attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    )

    assert_equal 2, scores.length
    scores.each { |s| assert_kind_of Float, s }
  end

  def test_score_returns_flat_scores
    reranker = OnnxRuby::Reranker.new(RERANKER_MODEL)

    scores = reranker.score(
      input_ids: [[1, 2, 3]],
      attention_mask: [[1, 1, 1]]
    )

    assert_equal 1, scores.length
    assert_kind_of Float, scores.first
  end

  def test_rerank_requires_tokenizer
    reranker = OnnxRuby::Reranker.new(RERANKER_MODEL)

    assert_raises(OnnxRuby::Error) do
      reranker.rerank("query", ["doc1", "doc2"])
    end
  end

  def test_session_accessible
    reranker = OnnxRuby::Reranker.new(RERANKER_MODEL)
    assert_kind_of OnnxRuby::Session, reranker.session
  end
end
