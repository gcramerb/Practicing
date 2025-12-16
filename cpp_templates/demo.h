#ifndef ARRAY_H
#define ARRAY_H

#include <cassert>

template <typename T> // added
class Array
{
private:
    int m_length{};
    T* m_data{}; // changed type to T

public:

    Array(int length);

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    ~Array();

    void erase();

    // templated operator[] function defined below
    //T& operator[](int index); // now returns a T&

    int getLength() const ;
};

// Definition of Array<T>::operator[] moved into Array.cpp below

#endif