#include "demo.h"
#include <cassert>

template <typename T> // added
Array::Array(int length)
{
    assert(length > 0);
    m_data = new T[length]{}; // allocated an array of objects of type T
    m_length = length;
}
template <typename T> // added
Array::~Array()
{
    delete[] m_data;
}
template <typename T> // added
void Array::erase()
{
    delete[] m_data;
    // We need to make sure we set m_data to 0 here, otherwise it will
    // be left pointing at deallocated memory!
    m_data = nullptr;
    m_length = 0;
}

// templated operator[] function defined below
//T& Array::operator[](int index); // now returns a T&
template <typename T> // added
int Array::getLength() const { return m_length; }