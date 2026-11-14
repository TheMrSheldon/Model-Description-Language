// mdl-format: formats `.dlm` files (like clang-format). The style comes from `--style-file`, else from the closest
// `.mdl-format` file of each source, else the defaults (see mdl::format::FormatOptions).
#include <mdl/format.hpp>
#include <mdl/version.hpp>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
	std::optional<std::string> readFile(const std::filesystem::path& path) {
		std::ifstream in{path, std::ios::binary};
		if (!in)
			return std::nullopt;
		return std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
	}
} // namespace

int main(int argc, char* argv[]) {
	CLI::App app("mdl-format -- formats model description language files");
	app.set_version_flag("--version", std::string{mdl::version});
	std::vector<std::filesystem::path> files;
	bool inPlace = false, check = false;
	std::optional<std::filesystem::path> styleFile;
	std::string assumeFilename = "<stdin>";
	app.add_option("files", files, "The files to format; without files, stdin is formatted to stdout.");
	app.add_flag("-i,--in-place", inPlace, "Format the files in place.");
	app.add_flag(
			"--check,--dry-run", check,
			"Only check: report the files that are not formatted or break the naming rules of the style, and exit with "
			"1 "
			"if there are any."
	);
	app.add_option("--style-file", styleFile, "The style file (default: the closest .mdl-format of each file).")
			->check(CLI::ExistingFile);
	app.add_option("--assume-filename", assumeFilename, "The file name of stdin (for messages and the style file).");
	CLI11_PARSE(app, argc, argv);

	auto styleFor = [&](const std::filesystem::path& source) -> std::optional<mdl::format::FormatOptions> {
		auto file = styleFile ? styleFile : mdl::format::findStyleFile(source);
		if (!file)
			return mdl::format::FormatOptions{};
		auto style = mdl::format::loadStyle(*file);
		if (!style) {
			std::cerr << "mdl-format: error: " << style.error() << "\n";
			return std::nullopt;
		}
		return *style;
	};

	auto formatOne = [&](const std::string& source, const std::string& name,
						 const std::filesystem::path& stylePath) -> std::optional<mdl::format::FormatResult> {
		auto style = styleFor(stylePath);
		if (!style)
			return std::nullopt;
		auto result = mdl::format::format(source, *style, name);
		for (const auto& d : result.diagnostics)
			mdl::printDiagnostic(std::cerr, d);
		if (!result.ok)
			return std::nullopt;
		return result;
	};

	if (files.empty()) {
		std::string source{std::istreambuf_iterator<char>{std::cin}, std::istreambuf_iterator<char>{}};
		auto result = formatOne(source, assumeFilename, std::filesystem::current_path() / assumeFilename);
		if (!result) {
			std::cout << source;
			return 1;
		}
		if (check)
			return result->text == source && std::none_of(
													 result->diagnostics.begin(), result->diagnostics.end(),
													 [](const auto& d) { return d.flag == "naming"; }
											 )
						   ? 0
						   : 1;
		std::cout << result->text;
		return 0;
	}

	int status = 0;
	for (const auto& file : files) {
		auto source = readFile(file);
		if (!source) {
			std::cerr << "mdl-format: error: cannot read " << file.string() << "\n";
			status = 1;
			continue;
		}
		auto result = formatOne(*source, file.string(), file);
		if (!result) {
			status = 1;
			continue;
		}
		const bool namingWarnings =
				std::any_of(result->diagnostics.begin(), result->diagnostics.end(), [](const auto& d) {
					return d.flag == "naming";
				});
		if (check) {
			if (result->text != *source) {
				std::cerr << file.string() << ": not formatted\n";
				status = 1;
			}
			if (namingWarnings)
				status = 1;
		} else if (inPlace) {
			if (result->text != *source)
				std::ofstream{file, std::ios::binary} << result->text;
		} else {
			std::cout << result->text;
		}
	}
	return status;
}
