# frozen_string_literal: true

module OnnxRuby
  class Reranker
    include TokenizerSupport

    attr_reader :session

    def initialize(model_path, tokenizer: nil, **session_opts)
      @session = Session.new(model_path, **session_opts)
      @tokenizer = resolve_tokenizer(tokenizer)
    end

    # Rerank documents by relevance to a query
    # @param query [String] the query text (requires tokenizer)
    # @param documents [Array<String>] documents to rerank
    # @return [Array<Hash>] sorted array of { document:, score:, index: }
    def rerank(query, documents)
      raise Error, "tokenizer is required for reranking" unless @tokenizer

      pairs = documents.map { |doc| [query, doc] }
      scores = score_pairs(pairs)

      documents.each_with_index.map do |doc, i|
        { document: doc, score: scores[i], index: i }
      end.sort_by { |r| -r[:score] }
    end

    # Score query-document pairs with pre-tokenized inputs
    # @param input_ids [Array<Array<Integer>>] batch of token ID sequences
    # @param attention_mask [Array<Array<Integer>>] batch of attention masks
    # @return [Array<Float>] relevance scores
    def score(input_ids:, attention_mask:)
      feed = build_feed(input_ids, attention_mask)
      result = @session.run(feed)
      raw_scores = find_output(result, %w[scores logits output])
      raw_scores.map { |row| row.is_a?(Array) ? row.first : row }
    end

    private

    def score_pairs(pairs)
      if @tokenizer.respond_to?(:encode_batch)
        encodings = @tokenizer.encode_batch(pairs)
        ids = encodings.map(&:ids)
        masks = encodings.map(&:attention_mask)
      else
        encodings = pairs.map { |pair| @tokenizer.encode(*pair) }
        ids = encodings.map(&:ids)
        masks = encodings.map(&:attention_mask)
      end

      max_len = ids.map(&:length).max
      ids = ids.map { |row| row + Array.new(max_len - row.length, 0) }
      masks = masks.map { |row| row + Array.new(max_len - row.length, 0) }

      feed = build_feed(ids, masks)
      result = @session.run(feed)
      raw_scores = find_output(result, %w[scores logits output])
      raw_scores.map { |row| row.is_a?(Array) ? row.first : row }
    end

    def build_feed(ids, masks)
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
  end
end
