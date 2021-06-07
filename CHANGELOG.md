
# Change Log
All notable changes to this project will be documented in this file.

## [0.3.0] 2021-06-07

This version introduces a persitence layer for encoders.

### Added
- Serialization for encoders
  
### Changed
- Interface of encoders
- Minor things in the library

### Fixed
- Some issues with serialization of online, network


## [0.2.5] 2021-06-06

With this version we introduce encoders (automatic encoders) to gopher-learn.
You now can reduce large float slice inputs or encode your string input right away.

### Added
- Encoders for float slices and string input.
- With encoders large float input can be reduced using Spearman.
- Also with encoders strings can be encoded as ngrams and dictionary (Topic modelling to come soon)
  
### Changed
- Relocated the neural net from neural package into an own package called net

### Fixed
- Nothing here


## [0.2] - 2021-05-09
 
Introducing online learning.
 
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
