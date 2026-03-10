# frozen_string_literal: true

module OnnxRuby
  module TokenizerSupport
    private

    def resolve_tokenizer(tokenizer)
      case tokenizer
      when String
        require "tokenizers"
        Tokenizers::Tokenizer.from_pretrained(tokenizer.to_s)
      when nil
        nil
      else
        tokenizer
      end
    rescue LoadError
      raise OnnxRuby::Error,
        "tokenizers gem required for text inputs. Install: gem install tokenizers"
    end
  end
end
