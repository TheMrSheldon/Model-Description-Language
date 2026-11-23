#include "./protocol.hpp"
#include "./utf8.hpp"

#include <cctype>
#include <cstdint>

std::optional<mdls::json> mdls::readMessage(std::istream& in) {
	size_t length = 0;
	bool haveLength = false;
	std::string line;
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		if (line.empty()) {
			if (!haveLength)
				continue; // tolerate stray blank lines
			std::string body(length, '\0');
			in.read(body.data(), static_cast<std::streamsize>(length));
			if (static_cast<size_t>(in.gcount()) != length)
				return std::nullopt;
			return json::parse(body, nullptr, /*allow_exceptions=*/false);
		}
		constexpr std::string_view header = "Content-Length:";
		if (line.size() > header.size() && std::equal(header.begin(), header.end(), line.begin(), [](char a, char b) {
				return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
			})) {
			const std::string value = line.substr(header.size());
			size_t used = 0;
			unsigned long long parsed = 0;
			try {
				parsed = std::stoull(value, &used);
			} catch (const std::exception&) {
				throw ProtocolError("invalid Content-Length: " + value);
			}
			if (value.find_first_not_of(" \t", used) != std::string::npos || value.find('-') != std::string::npos)
				throw ProtocolError("invalid Content-Length: " + value);
			if (parsed > maxMessageSize)
				throw ProtocolError("message too large: " + value + " bytes");
			length = static_cast<size_t>(parsed);
			haveLength = true;
		}
	}
	return std::nullopt;
}

void mdls::writeMessage(std::ostream& out, const json& message) {
	const std::string body = message.dump(-1, ' ', false, json::error_handler_t::replace);
	out << "Content-Length: " << body.size() << "\r\n\r\n" << body;
	out.flush();
}

std::optional<std::filesystem::path> mdls::uriToPath(std::string_view uri) {
	constexpr std::string_view scheme = "file://";
	if (uri.substr(0, scheme.size()) != scheme)
		return std::nullopt;
	std::string_view rest = uri.substr(scheme.size());
	std::string decoded;
	for (size_t i = 0; i < rest.size(); ++i) {
		if (rest[i] == '%' && i + 2 < rest.size() && std::isxdigit(static_cast<unsigned char>(rest[i + 1])) &&
			std::isxdigit(static_cast<unsigned char>(rest[i + 2]))) {
			decoded += static_cast<char>(std::stoi(std::string{rest.substr(i + 1, 2)}, nullptr, 16));
			i += 2;
		} else {
			decoded += rest[i];
		}
	}
#if defined(_WIN32)
	// file:///c:/x -> c:/x
	if (decoded.size() >= 3 && decoded[0] == '/' && decoded[2] == ':')
		decoded.erase(0, 1);
#endif
	return std::filesystem::path{decoded}.lexically_normal();
}

std::string mdls::pathToUri(const std::filesystem::path& path) {
	std::string p = path.generic_string();
	std::string uri = "file://";
	if (p.empty() || p[0] != '/')
		uri += '/'; // Windows: file:///c:/...
	static constexpr char hex[] = "0123456789ABCDEF";
	for (unsigned char c : p) {
		if (std::isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' || c == '~' || c == ':')
			uri += static_cast<char>(c);
		else {
			uri += '%';
			uri += hex[c >> 4];
			uri += hex[c & 15];
		}
	}
	return uri;
}

mdls::LineIndex::LineIndex(std::string_view text) {
	size_t start = 0;
	while (true) {
		const size_t nl = text.find('\n', start);
		std::string line{text.substr(start, nl == std::string_view::npos ? std::string_view::npos : nl - start)};
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		lines.push_back(std::move(line));
		if (nl == std::string_view::npos)
			break;
		start = nl + 1;
	}
}

std::string_view mdls::LineIndex::line(size_t line) const {
	return line < lines.size() ? std::string_view{lines[line]} : std::string_view{};
}

size_t mdls::LineIndex::toUtf16(size_t line, size_t column) const {
	const std::string_view text = this->line(line);
	size_t units = 0, points = 0;
	for (size_t i = 0; i < text.size() && points < column; ++points) {
		units += utf8::utf16Units(text.substr(i));
		i += utf8::sequenceLength(text.substr(i));
	}
	return units + (column > points ? column - points : 0); // beyond the end: one unit per code point
}

size_t mdls::LineIndex::fromUtf16(size_t line, size_t column) const {
	const std::string_view text = this->line(line);
	size_t units = 0, points = 0;
	for (size_t i = 0; i < text.size() && units < column; ++points) {
		units += utf8::utf16Units(text.substr(i));
		i += utf8::sequenceLength(text.substr(i));
	}
	return points + (column > units ? column - units : 0);
}
