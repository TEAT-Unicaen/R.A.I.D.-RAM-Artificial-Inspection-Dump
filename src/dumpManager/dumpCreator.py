"""
This module contains the DumpCreator class, which is responsible for creating dumps of data, imitating RAM Dumps.
These dumps will be represented by seeds, which can be used to regenerate the same dumps when needed.
"""

import os
import random

class DumpCreator():

    seed: int

    _size: int
    _fileTypes: list[str]
    _filePath: str
    _fragmentedBuffer: bool
    _failRate: float

    """
    __init__ Method for DumpCreator class:
        @param size: Size of the dump in MB
        @param fileTypes: List of file types to include in the dump. If set to None, defaults to all types.
        @param filePath: Path to save the dump files. If None, uses default path.
        @param fragmentedBuffer: Whether to create a fragmented memory buffer
        @param seed: Seed for random generation to ensure reproducibility
        @param failRate: Percentage of failed attempts to fit files into memory buffer before stopping; Expressed as a percentage (0-100)
    """
    def __init__(self, size = 10, fileTypes = ["pdf", "jpg", "txt", "png"], filePath = "dataset/train", fragmentedBuffer = False, seed = None, failRate = 5):
        self._size = size
        self._fileTypes = fileTypes
        self._filePath = filePath
        self._fragmentedBuffer = fragmentedBuffer
        self._seed = seed
        self._failRate = failRate

        self.createDump(size, fileTypes, seed, failRate)


    def createDump(self, size, fileTypes, seed, failRate = 5):
        print(f"Creating a dump of size {size}MB with file types {fileTypes}.")

        memoryBuffer = bytearray(size * 1024 * 1024) # Size in MB 
        files = self.loadFilesByType(fileTypes, self._filePath)
        
        #Shuffle files to ensure randomness, and restore state after
        seedSaved = random.getstate()
        random.seed(seed)
        random.shuffle(files)
        random.setstate(seedSaved)

        fails = 0
        for file in files:
            with open(file, "rb") as f:
                fileData = f.read()
                fileSize = len(fileData)

                #If we fail a number of times equivalent to failRate% of the total files count, we can assume the buffer is full
                failRate = failRate / 100
                if fails > len(files) * failRate:
                    print("Too many failed attempts to fit files into memory buffer. We can assume its nearly full. Stopping.")
                    break

                if fileSize > len(memoryBuffer):
                    fails += 1
                    continue  # Skip files that are too large to fit

                if self._fragmentedBuffer:
                    print("Fragmented buffer not yet implemented.")
                    pass
                else:
                    # Contiguous buffer: find the first fit
                    startIndex = memoryBuffer.find(bytearray(fileSize))
                    if startIndex == -1:
                        fails += 1
                        continue  # No space available

                # Place file data into memory buffer
                memoryBuffer[startIndex:startIndex + fileSize] = fileData
                fails = 0

        #TODO : Finish it maybe?


    def loadFilesByType(self, fileTypes, filePath):
        print(f"Loading files of type: {fileTypes} from path: {filePath}")

        #Verify the path exists
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"The specified path does not exist: {filePath}")

        #Load files from path that match the fileTypes
        matching_files = []
        for root, dirs, files in os.walk(filePath):
            for file in files:
                if file.split('.')[-1] in fileTypes:
                    matching_files.append(os.path.join(root, file))

        return matching_files
