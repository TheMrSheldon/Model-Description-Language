#ifndef MDL_NAMESPACE_HPP
#define MDL_NAMESPACE_HPP

#include <map>
#include <string>

#include <iostream>

namespace mdl {

    class Namespace final {
    private:
        
    public:
        template<typename T>
        void put(std::string_view name, T&& obj) {
            std::cout << "Adding " << name << std::endl;
        }

        template<typename T>
        const T& get(std::string_view name) {
            std::cout << "Fetching " << name << std::endl;
        }
    };

}

#endif