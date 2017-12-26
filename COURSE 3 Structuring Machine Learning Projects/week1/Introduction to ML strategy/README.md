## Why ML Strategy

### Ideas

- Collect more data
- Collect more diverse traning set
- Train algorithm longer with gradient descent
- Try Adam instread of gradient descent
- Try bigger / smaller network
- Try dropout
- Add L2 regularization
- Network architecture
  - activation functions
  - \# of hidden units

## Orthogonalization

Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects to other components of the system. It becomes easier to verify the algorithms independently from one another, it reduces testing and development time. 

### Orthogonalization in supervised learning
When a supervised learning system is design, these are the 4assumptions that needs to be true and orthogonal. 

1. Fit training set well in costfunction 

  - If it doesn’t fit well, the use of a bigger neural network or switching to a better optimization algorithm might help. 

2. Fit development set well oncost function 

  - If it doesn’t fit well,regularization or using bigger training set might help. 

3. Fit test set well on costfunction 

  - If it doesn’t fit well, the use of a bigger development set might help 

4. Performs well in real world 

  - If it doesn’t perform well, the development test set is not set correctly or the cost function is notevaluating the right thing. 

