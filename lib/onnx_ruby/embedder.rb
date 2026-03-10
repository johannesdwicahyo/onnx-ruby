# frozen_string_literal: true

module OnnxRuby
  class Embedder
    include TokenizerSupport

    attr_reader :session

    def initialize(model_path, tokenizer: nil, normalize: true, **session_opts)
      @session = Session.new(model_path, **session_opts)
      @normalize = normalize
      @tokenizer = resolve_tokenizer(tokenizer)
    end

    # Embed a single text or pre-tokenized input
    # @param input [String, Hash] text string (requires tokenizer) or hash of input tensors
    # @return [Array<Float>] embedding vector
    def embed(input)
      embed_batch([input]).first
    end

    # Embed a batch of texts or pre-tokenized inputs
    # @param inputs [Array<String>, Array<Hash>] batch of texts or tensor hashes
    # @return [Array<Array<Float>>] array of embedding vectors
    def embed_batch(inputs)
      @_masks = nil
      feed = prepare_inputs(inputs)
      result = @session.run(feed)

      raw = find_output(result, %w[embeddings sentence_embedding output last_hidden_state])
      return [] if raw.nil? || raw.empty?

      # If output is 3D (batch, seq_len, dim) — do mean pooling
      embeddings = if raw.first.is_a?(Array) && raw.first.first.is_a?(Array)
                     mean_pool(raw, @_masks)
                   else
                     raw
                   end

      embeddings.map { |vec| @normalize ? l2_normalize(vec) : vec }
    end

    private

    def prepare_inputs(inputs)
      if inputs.first.is_a?(String)
        raise Error, "tokenizer is required for text inputs" unless @tokenizer

        tokenize_batch(inputs)
      elsif inputs.first.is_a?(Hash)
        # Merge hash inputs into batched arrays
        merge_input_hashes(inputs)
      else
        raise Error, "inputs must be Strings or Hashes"
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

      # Pad to max length
      max_len = ids.map(&:length).max
      ids = ids.map { |row| row + Array.new(max_len - row.length, 0) }
      masks = masks.map { |row| row + Array.new(max_len - row.length, 0) }

      build_feed(ids, masks)
    end

    def merge_input_hashes(hashes)
      result = {}
      hashes.first.each_key do |key|
        result[key] = hashes.map { |h| h[key] }
      end
      # Stash masks for mean pooling
      mask_key = result.keys.find { |k| k.to_s.include?("mask") || k.to_s.include?("attention") }
      @_masks = result[mask_key] if mask_key
      result
    end

    def build_feed(ids, masks)
      input_names = @session.inputs.map { |i| i[:name] }
      raise OnnxRuby::Error, "Model has no input names" if input_names.empty?

      feed = {}
      feed[input_names.find { |n| n.include?("input_id") } || input_names[0]] = ids
      mask_name = input_names.find { |n| n.include?("mask") || n.include?("attention") }
      feed[mask_name] = masks if mask_name
      # Supply token_type_ids (zeros) if the model expects it
      tti_name = input_names.find { |n| n.include?("token_type") }
      feed[tti_name] = ids.map { |row| Array.new(row.length, 0) } if tti_name
      @_masks = masks  # stash for mean pooling
      feed
    end

    def find_output(result, candidate_names)
      candidate_names.each { |name| return result[name] if result.key?(name) }
      result.values.first
    end

    # Mean pooling over token embeddings, masked by attention_mask
    def mean_pool(hidden_states, masks)
      hidden_states.each_with_index.map do |tokens, batch_idx|
        return [] if tokens.nil? || tokens.empty? || tokens.first.nil?

        mask = masks && masks[batch_idx]
        dim = tokens.first.length
        sum = Array.new(dim, 0.0)
        count = 0.0

        tokens.each_with_index do |token_vec, tok_idx|
          w = (mask && mask[tok_idx]) ? mask[tok_idx].to_f : 1.0
          next if w.zero?

          count += w
          token_vec.each_with_index { |v, d| sum[d] += v * w }
        end

        count = 1.0 if count.zero?
        sum.map { |v| v / count }
      end
    end

    def l2_normalize(vec)
      norm = Math.sqrt(vec.sum { |v| v * v })
      return vec if norm.zero?

      vec.map { |v| v / norm }
    end
  end
end
