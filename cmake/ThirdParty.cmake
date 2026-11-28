# The licenses of the third-party software in the binaries that we distribute (Apache-2.0, BSD and MIT licenses require
# that the license text and the notices accompany the binaries).
#
# The places that add a dependency register it:
#
#   mdl_third_party(<name> COMPONENTS <component>... LICENSE_FILES <file>... [VERSION <v>] [SPDX <id>] [URL <url>] [NOTE <text>])
#
# `COMPONENTS` are the install components whose files contain the software (compiled in, linked statically, or its
# headers). mdl_install_third_party_notices() (Packaging.cmake) writes one file THIRD-PARTY-NOTICES.txt per component
# and installs it to share/doc/mdl-<component>/ (like the Debian doc directory of the package of that component; one
# file per package, so that no two packages install the same file).
#
# A missing license file is a warning; with MDL_REQUIRE_THIRD_PARTY_LICENSES (the release builds) an error.
option(MDL_REQUIRE_THIRD_PARTY_LICENSES "Fail if the license of a third-party dependency cannot be found" OFF)

function(mdl_third_party name)
	cmake_parse_arguments(PARSE_ARGV 1 ARG "" "VERSION;SPDX;URL;NOTE" "COMPONENTS;LICENSE_FILES")
	set(files)
	foreach(file IN LISTS ARG_LICENSE_FILES)
		# (a pattern selects the first match)
		file(GLOB matches "${file}")
		list(SORT matches)
		if(matches)
			list(APPEND files ${matches})
		elseif(MDL_REQUIRE_THIRD_PARTY_LICENSES)
			message(FATAL_ERROR "third-party license of ${name}: ${file} not found")
		else()
			message(WARNING "third-party license of ${name}: ${file} not found")
		endif()
	endforeach()
	set_property(GLOBAL APPEND PROPERTY MDL_THIRD_PARTY "${name}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_COMPONENTS "${ARG_COMPONENTS}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_FILES "${files}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_VERSION "${ARG_VERSION}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_SPDX "${ARG_SPDX}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_URL "${ARG_URL}")
	set_property(GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_NOTE "${ARG_NOTE}")
endfunction()

# The first existing file of the patterns (e.g. where distributions put the license of LLVM).
function(mdl_find_license_file var)
	foreach(pattern IN LISTS ARGN)
		file(GLOB matches "${pattern}")
		list(SORT matches)
		if(matches)
			list(GET matches 0 first)
			set(${var} "${first}" PARENT_SCOPE)
			return()
		endif()
	endforeach()
	set(${var} "" PARENT_SCOPE)
endfunction()

# Writes and installs the notices of the components.
function(mdl_install_third_party_notices)
	get_property(names GLOBAL PROPERTY MDL_THIRD_PARTY)
	foreach(component IN LISTS ARGN)
		string(CONCAT text "Third-party software in the MDL package mdl-${component}\n"
			"===================================================\n\n"
			"The binaries of this package contain or are built with the following software; "
			"its license and notices follow.\n")
		set(count 0)
		foreach(name IN LISTS names)
			get_property(components GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_COMPONENTS)
			if(NOT component IN_LIST components)
				continue()
			endif()
			math(EXPR count "${count} + 1")
			get_property(files GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_FILES)
			get_property(version GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_VERSION)
			if(version)
				set(version " ${version}")
			endif()
			get_property(spdx GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_SPDX)
			get_property(url GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_URL)
			get_property(note GLOBAL PROPERTY MDL_THIRD_PARTY_${name}_NOTE)
			string(APPEND text "\n\n################################################################################\n"
				"# ${name}${version}\n")
			if(url)
				string(APPEND text "# ${url}\n")
			endif()
			if(spdx)
				string(APPEND text "# License: ${spdx}\n")
			endif()
			string(APPEND text "################################################################################\n")
			if(note)
				string(APPEND text "\n${note}\n")
			endif()
			foreach(file IN LISTS files)
				file(READ "${file}" content)
				get_filename_component(file_name "${file}" NAME)
				string(APPEND text "\n--- ${file_name} ---\n\n${content}\n")
			endforeach()
		endforeach()
		if(count EQUAL 0)
			continue()
		endif()
		set(output "${CMAKE_BINARY_DIR}/third-party/${component}/THIRD-PARTY-NOTICES.txt")
		file(WRITE "${output}" "${text}")
		install(FILES "${output}" DESTINATION share/doc/mdl-${component} COMPONENT ${component})
	endforeach()
endfunction()
