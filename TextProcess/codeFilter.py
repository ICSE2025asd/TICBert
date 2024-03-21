from TextProcess import TextProcessConfig
import re


def javaCodeFilter(s):
    """
    Public. Find all JAVA code in the string based on the regular expression
    :param s: string
    :return: codeRegions: list, list of code regions, each code region contains 4 attributes, namely start index,
               end index, tag, code
    """
    codeRegions = []
    for keyword in TextProcessConfig.CodePattern.JAVACODEPATTERN.keys():
        pattern = TextProcessConfig.CodePattern.JAVACODEPATTERN[keyword]
        patternOptions = TextProcessConfig.CodePattern.JAVACODEPATTERNOPTIONS[keyword]
        if patternOptions == "MATCH":
            for match in re.finditer(pattern, s, flags=re.DOTALL):
                indexStart = match.start()
                indexEnd = match.end()
                offset = findmatch(s, '{', '}', indexEnd-1)
                codeRegion = [indexStart, indexEnd + offset, keyword, s[indexStart: indexEnd + offset]]
                codeRegions.append(codeRegion)
        else:
            for match in re.finditer(pattern, s, flags=re.DOTALL):
                indexStart = match.start()
                indexEnd = match.end()
                codeRegion = [indexStart, indexEnd, keyword, s[indexStart: indexEnd]]
                codeRegions.append(codeRegion)
    codeRegions = makeMinimalSet(codeRegions)
    return codeRegions


def findmatch(s, opening, closing, start):
    """
    findMatch() returns the offset where the next closing is found. If not found return 0. Can handle nesting.
    :param s: string
    :param opening: char
    :param closing: char
    :param start: int, Decide where to start searching
    :return: position: int
    """
    s = s[start:]
    level = 0
    position = 0
    for c in s:
        position = position + 1
        if c == opening:
            level = level + 1
        if c == closing:
            level = level - 1
            if level == 0:
                return position

    return 0


def makeMinimalSet(codeRegions):
    """
    De duplication according to the start position and end position.
    :param codeRegions: list, list of code regions, each code region contains 4 attributes, namely start index,
             end index, tag, code
    :return: miniSet: list, list of code regions, each code region contains 4 attributes, namely start index,
               end index, tag, code
    """
    codeRegions.sort(key=lambda x: x[0])
    miniSet = []
    for i in range(0, len(codeRegions)):
        thisRegion = codeRegions[i]
        isContained = False
        if thisRegion[2] == 'multicomment':
            isContained = False
        elif thisRegion[2] == 'singlecomment':
            for j in range(0, i):
                thatRegion = codeRegions[j]
                if thatRegion[2] == 'multicomment' and thatRegion[1] >= thisRegion[1]:
                    isContained = True
                    break
        else:
            for j in range(0, i):
                thatRegion = codeRegions[j]
                if thatRegion[1] >= thisRegion[1]:
                    isContained = True
                    break
        if not isContained:
            if thisRegion[2] != 'multicomment' and thisRegion[2] != 'singlecomment':
                for thatRegion in miniSet:
                    if thatRegion[1] >= thisRegion[0]:
                        thisRegion[3] = thisRegion[3][thatRegion[1] - thisRegion[0]:]
                        thisRegion[0] = thatRegion[1]
            miniSet.append(thisRegion)
    return miniSet
