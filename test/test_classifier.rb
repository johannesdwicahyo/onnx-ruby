# frozen_string_literal: true

require "test_helper"

class TestClassifier < Minitest::Test
  include TestHelper

  CLASSIFIER_MODEL = File.join(MODELS_DIR, "classifier.onnx")
  LABELS = %w[greeting farewell question command].freeze

  def setup
    skip "classifier.onnx not found" unless File.exist?(CLASSIFIER_MODEL)
  end

  def test_predict_with_features
    classifier = OnnxRuby::Classifier.new(CLASSIFIER_MODEL, labels: LABELS)

    result = classifier.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    assert_kind_of Hash, result
    assert LABELS.include?(result[:label])
    assert_kind_of Float, result[:score]
    assert result[:score] > 0.0
    assert result[:score] <= 1.0
    assert_equal 4, result[:scores].length
  end

  def test_predict_batch
    classifier = OnnxRuby::Classifier.new(CLASSIFIER_MODEL, labels: LABELS)

    results = classifier.predict_batch([
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ])

    assert_equal 2, results.length
    results.each do |r|
      assert LABELS.include?(r[:label])
      assert_kind_of Float, r[:score]
    end
  end

  def test_predict_without_labels
    classifier = OnnxRuby::Classifier.new(CLASSIFIER_MODEL)

    result = classifier.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    assert_kind_of Integer, result[:label]  # index when no labels
    assert result[:label] >= 0
    assert result[:label] < 4
  end

  def test_scores_sum_to_one
    classifier = OnnxRuby::Classifier.new(CLASSIFIER_MODEL, labels: LABELS)

    result = classifier.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    assert_in_delta 1.0, result[:scores].sum, 0.001
  end

  def test_requires_tokenizer_for_strings
    classifier = OnnxRuby::Classifier.new(CLASSIFIER_MODEL)

    assert_raises(OnnxRuby::Error) do
      classifier.predict("hello")
    end
  end
end
