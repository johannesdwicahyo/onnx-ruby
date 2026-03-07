# frozen_string_literal: true

require "rake/extensiontask"
require "rake/testtask"

Rake::ExtensionTask.new("onnx_ruby_ext") do |ext|
  ext.lib_dir = "lib/onnx_ruby"
  ext.ext_dir = "ext/onnx_ruby"
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/test_*.rb"]
end

task default: %i[compile test]
