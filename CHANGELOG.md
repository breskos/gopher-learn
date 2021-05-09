
# Change Log
All notable changes to this project will be documented in this file.

## [0.2] - 2021-05-09
 
Here we write upgrading notes for brands. It's a team effort to make them as
straightforward as possible.
 
### Added
- Config for online learner to control learning behavior - easily inject your own config
- Config for engine learner to control learning behavior - easily inject your own config
- Wrote Comments to every function to make everything easier to understand
- Online learner can now be serialized to disk using persist.OnlineToFile() and load using persist.OnlineFromFile()
  
### Changed
- Refactoring of Criterion handling. It is now part of the neural package (usage e.g.: neural.Distance)
- Reworked some of the examples

### Fixed
- Fixed regression example, did not work correctly
