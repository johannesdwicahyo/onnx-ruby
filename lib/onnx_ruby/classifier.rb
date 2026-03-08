# frozen_string_literal: true

module OnnxRuby
  class Classifier
    attr_reader :session, :labels

    def initialize(model_path, tokenizer: nil, labels: nil, **session_opts)
      @session = Session.new(model_path, **session_opts)
      @labels = labels
      @tokenizer = resolve_tokenizer(tokenizer)
    end

    # Classify a single input
    # @param input [String, Array<Float>] text (requires tokenizer) or feature vector
    # @return [Hash] { label:, score:, scores: }
    def predict(input)
      predict_batch([input]).first
    end

    # Classify a batch of inputs
    # @param inputs [Array<String>, Array<Array<Float>>] batch of texts or feature vectors
    # @return [Array<Hash>] array of { label:, score:, scores: }
    def predict_batch(inputs)
      feed = prepare_inputs(inputs)
      result = @session.run(feed)

      logits = find_output(result, %w[logits output probabilities scores])

      logits.map { |row| format_prediction(row) }
    end

    private

    def resolve_tokenizer(tokenizer)
      return nil if tokenizer.nil?

      if tokenizer.respond_to?(:encode)
        tokenizer
      else
        begin
          require "tokenizers"
          Tokenizers::Tokenizer.from_pretrained(tokenizer.to_s)
        rescue LoadError
          raise Error, "tokenizer-ruby gem is required for text tokenization. " \
                       "Install with: gem install tokenizers"
        end
      end
    end

    def prepare_inputs(inputs)
      if inputs.first.is_a?(String)
        raise Error, "tokenizer is required for text inputs" unless @tokenizer

        tokenize_batch(inputs)
      elsif inputs.first.is_a?(Array)
        # Raw feature vectors
        input_name = @session.inputs.first[:name]
        { input_name => inputs }
      else
        raise Error, "inputs must be Strings or Arrays"
      end
    end

    def tokenize_batch(texts)
      if @tokenizer.respond_to?(:encode_batch)
        encodings = @tokenizer.encode_batch(texts)
        ids = encodings.map(&:ids)
        masks = encodings.map(&:attention_mask)
      else
        encodings = texts.map { |t| @tokenizer.encode(t) }
        ids = encodings.map(&:ids)
        masks = encodings.map(&:attention_mask)
      end

      max_len = ids.map(&:length).max
      ids = ids.map { |row| row + Array.new(max_len - row.length, 0) }
      masks = masks.map { |row| row + Array.new(max_len - row.length, 0) }

      input_names = @session.inputs.map { |i| i[:name] }
      feed = {}
      feed[input_names.find { |n| n.include?("input_id") } || input_names[0]] = ids
      mask_name = input_names.find { |n| n.include?("mask") || n.include?("attention") }
      feed[mask_name] = masks if mask_name
      feed
    end

    def find_output(result, candidate_names)
      candidate_names.each { |name| return result[name] if result.key?(name) }
      result.values.first
    end

    def format_prediction(logits_row)
      probs = softmax(logits_row)
      max_idx = probs.each_with_index.max_by(&:first).last
      label = @labels ? @labels[max_idx] : max_idx

      { label: label, score: probs[max_idx], scores: probs }
    end

    def softmax(logits)
      max_val = logits.max
      exps = logits.map { |v| Math.exp(v - max_val) }
      sum = exps.sum
      exps.map { |v| v / sum }
    end
  end
end
