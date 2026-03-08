# frozen_string_literal: true

require "test_helper"

class TestEmbedder < Minitest::Test
  include TestHelper

  EMBEDDING_MODEL = File.join(MODELS_DIR, "embedding.onnx")

  def setup
    skip "embedding.onnx not found" unless File.exist?(EMBEDDING_MODEL)
  end

  def test_embed_with_raw_inputs
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL)

    result = embedder.embed({
      "input_ids" => [1, 2, 3, 4, 5, 6],
      "attention_mask" => [1, 1, 1, 1, 1, 1]
    })

    assert_kind_of Array, result
    assert_equal 8, result.length  # embed_dim = 8
    assert_kind_of Float, result.first
  end

  def test_embed_batch_with_raw_inputs
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL)

    results = embedder.embed_batch([
      { "input_ids" => [1, 2, 3, 4, 5, 6], "attention_mask" => [1, 1, 1, 1, 1, 1] },
      { "input_ids" => [7, 8, 9, 10, 11, 12], "attention_mask" => [1, 1, 1, 1, 1, 0] }
    ])

    assert_equal 2, results.length
    assert_equal 8, results[0].length
    assert_equal 8, results[1].length
  end

  def test_normalized_embeddings
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL, normalize: true)

    result = embedder.embed({
      "input_ids" => [1, 2, 3, 4, 5, 6],
      "attention_mask" => [1, 1, 1, 1, 1, 1]
    })

    norm = Math.sqrt(result.sum { |v| v * v })
    assert_in_delta 1.0, norm, 0.001
  end

  def test_unnormalized_embeddings
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL, normalize: false)

    result = embedder.embed({
      "input_ids" => [1, 2, 3, 4, 5, 6],
      "attention_mask" => [1, 1, 1, 1, 1, 1]
    })

    assert_kind_of Array, result
    assert_equal 8, result.length
  end

  def test_requires_tokenizer_for_strings
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL)

    assert_raises(OnnxRuby::Error) do
      embedder.embed("hello world")
    end
  end

  def test_session_accessible
    embedder = OnnxRuby::Embedder.new(EMBEDDING_MODEL)
    assert_kind_of OnnxRuby::Session, embedder.session
  end
end
