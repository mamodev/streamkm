#pragma once 

#include <concepts>

namespace streamkm
{
    template <typename T>
    concept MovableNotCopyable = std::movable<T> && !std::copy_constructible<T>;

    
} // namespace streamkm
