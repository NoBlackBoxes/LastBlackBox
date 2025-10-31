# Systems : Multi-Tasking
Back when computers were very expensive mainframes, some clever software engineers developed strategies to share the computer's resources between multiple users, running multiple different tasks, at the same time. These "scheduling" programs still form the basis of our modern **multi-tasking** operating systems.

## [Video](https://vimeo.com/1036086160)

## Concepts
- Multi-user paradigm (origin)
- Context switch (quickly) between users/tasks (scheduler)
- Manage each user's/task's allocated memory resources
- What about multiple-cores? (still share memory)
- Security issues
- Cons: No timing guarantees

## Lesson

- **TASK**: Develop a multi-tasking "scheduling" program for your microcontroller.
> Changing the timescale of one task should not affect the other.