require "mkmf-rice"
require "fileutils"
require "tmpdir"

ORT_VERSION = "1.24.3"

def detect_platform
  os = RbConfig::CONFIG["host_os"]
  cpu = RbConfig::CONFIG["host_cpu"]

  case os
  when /darwin/
    cpu =~ /arm64|aarch64/ ? "osx-arm64" : "osx-x86_64"
  when /linux/
    cpu =~ /aarch64|arm64/ ? "linux-aarch64" : "linux-x64"
  else
    abort "Unsupported OS: #{os}"
  end
end

def download_onnxruntime(dest_dir)
  platform = detect_platform
  filename = "onnxruntime-#{platform}-#{ORT_VERSION}.tgz"
  url = "https://github.com/microsoft/onnxruntime/releases/download/v#{ORT_VERSION}/#{filename}"
  tmp_file = File.join(Dir.tmpdir, filename)

  unless File.exist?(tmp_file)
    puts "Downloading ONNX Runtime v#{ORT_VERSION} for #{platform}..."
    system("curl", "-fSL", url, "-o", tmp_file) or
      abort "Failed to download ONNX Runtime from #{url}"
  end

  FileUtils.mkdir_p(dest_dir)
  puts "Extracting to #{dest_dir}..."
  system("tar", "xzf", tmp_file, "-C", dest_dir, "--strip-components=1") or
    abort "Failed to extract ONNX Runtime"

  dest_dir
end

# Find ONNX Runtime
ort_dir = ENV["ONNX_RUNTIME_DIR"]

unless ort_dir
  # Check for bundled copy
  bundled_dir = File.join(__dir__, "onnxruntime")
  if File.exist?(File.join(bundled_dir, "include", "onnxruntime_cxx_api.h"))
    ort_dir = bundled_dir
  else
    ort_dir = download_onnxruntime(bundled_dir)
  end
end

ort_include = File.join(ort_dir, "include")
ort_lib = File.join(ort_dir, "lib")

abort "Cannot find ONNX Runtime headers in #{ort_include}" unless File.exist?(File.join(ort_include, "onnxruntime_cxx_api.h"))

$INCFLAGS << " -I#{ort_include}"
$LDFLAGS << " -L#{ort_lib}"
$libs << " -lonnxruntime"

# Set rpath so the shared library can be found at runtime
case RbConfig::CONFIG["host_os"]
when /darwin/
  $LDFLAGS << " -Wl,-rpath,#{ort_lib}"
when /linux/
  $LDFLAGS << " -Wl,-rpath,#{ort_lib}"
end

$CXXFLAGS = ($CXXFLAGS || "") + " -std=c++17"

have_header("onnxruntime_cxx_api.h") or abort "Cannot find onnxruntime_cxx_api.h"

create_makefile("onnx_ruby/onnx_ruby_ext")
