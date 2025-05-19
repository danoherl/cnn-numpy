# cnn_numpy

An implementation of a ConvNet layer framework, using only NumPy.

### Notes
- This is predictably not fast, and sacrifices many optimizations of libraries such as scikit-learn. The main purpose is to understand the inner workings of a CNN. 
### Future Improvements
- Introduce polymorphism from an abstract base "Layer" Class. Use ABC.
- Initialise weights not as zeros. Thinking He initialization.
- Pull input out of being class variable 
