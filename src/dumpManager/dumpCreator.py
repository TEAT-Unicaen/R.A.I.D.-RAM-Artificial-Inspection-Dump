import os
import random

def createDump(size_mb: int, fileTypes: list[str], seed: int, filePath: str, failRate = 5, export = True) -> bytearray:

    """
    Dump creator function that simulates loading files into RAM.
    
    :param size_mb: Description
    :type size_mb: int
    :param fileTypes: Description
    :type fileTypes: list[str]
    :param seed: Description
    :type seed: int
    :param filePath: Description
    :type filePath: str
    :param failRate: Description
    :param export: Description
    :return: Description
    :rtype: bytearray
    """

    ram = bytearray(size_mb * 1024 * 1024)
    ram_size = len(ram)
    offset = 0
    rng = random.Random(seed)

    files = loadFilesByType(fileTypes, filePath)
    rng.shuffle(files)

    currentFailCount = 0

    filePosition = []

    for file in files:
        with open(file, 'rb') as f:
            data = f.read()
        file_size = len(data)

        if offset + file_size > ram_size:
            print(f"Not enough space to load file: {file}")
            currentFailCount += 1
            if currentFailCount > failRate: break
            continue
        
        filePosition.append((file, offset, offset + file_size))
        ram[offset:offset + file_size] = data
        offset += file_size

    if export:
        output = 'ram_dump.bin'
        with open(output, 'wb') as f:
            f.write(ram)
        print(f"Dump created and saved to {output}")
    return ram, filePosition

def loadFilesByType(fileTypes: list[str], filePath: str) -> list[str]:

    """
    Load files of specified types from a given directory.
    
    :param fileTypes: Description
    :type fileTypes: list[str]
    :param filePath: Description
    :type filePath: str
    :return: Description
    :rtype: list[str]
    """

    if not os.path.exists(filePath):
        raise FileNotFoundError(f"The specified path does not exist: {filePath}")

    matching_files = []
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file.split('.')[-1] in fileTypes:
                matching_files.append(os.path.join(root, file))

    return matching_files


if __name__ == "__main__":
    # Example usage
    size_mb = 100  # Size of the dump in MB
    fileTypes = ['txt']  # File types to include
    seed = 64  # Seed for randomness
    filePath = './data'  # Directory to load files from
    failRate = 5  # Number of allowed failures

    ram, filePosition = createDump(size_mb, fileTypes, seed, filePath, failRate, export=True)