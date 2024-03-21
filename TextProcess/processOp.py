from enum import Enum, unique

import pandas as pd
from nltk import regexp_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re

from TextProcess import stackTraceFilter
from TextProcess import codeFilter

from TextProcess import TextProcessConfig


def getUrls(s):
    """
    Get the location of the url in the string based on the regular expression
    :param s: string
    :return: urlLst: list, list of url-regions, each url-region contains 4 attributes, namely, start index, end index,
               tag, url
    """
    urlPattern = TextProcessConfig.RePattern.URLPATTERN
    urlLst = []
    for match in re.finditer(urlPattern, s, flags=re.DOTALL):
        urlLst.append([match.start(), match.end(), 'url', s[match.start(): match.end()]])
    return urlLst


def getColoredRegion(s):
    """
    Sometimes the reporter will color some paragraphs in the issue to highlight the key points. Get the location of the
    colored regions in the string based on the regular expression
    :param s: string
    :return: coloredRegion: list, list of colored regions, each colored region contains 4 attributes, namely start index,
               end index, tag, highlights
    """
    coloredRegion = []
    for match in re.finditer(TextProcessConfig.RePattern.COLOREDREGION, s, flags=re.DOTALL):
        tmp = s[match.start(): match.end()]
        tag = re.search(TextProcessConfig.RePattern.COLOREDTAG, tmp, flags=re.DOTALL)
        tagLength = tag.end() - tag.start()
        coloredRegion.append([match.start(), match.end(), tag.group(0), tmp[tagLength: -7]])
    return coloredRegion


def removeColorTag(s, regions):
    """
    Based on the position information obtained in getColoredRegion(s), remove the color tag from the string
    :param s: string
    :param regions: list, coloredRegion, example: [[297, 350, '{color:#ff0000}', ' any requests(include zkCli.sh)']]
    :return: s: string
    """
    regions.sort(key=lambda x: x[0], reverse=True)
    for r in regions:
        s = s[: r[0]] + ' ' + r[3] + ' ' + s[r[1]:]
    return s


def getExplicitCodeRegion(s):
    """
    Some code regions in the issue are explicitly indicated. Get the location of the code regions in the string based
    on the regular expression
    :param s: string
    :return: codeRegion: list, list of code regions, each code region contains 4 attributes, namely, start index,
               end index, tag, code
    """
    codeRegion = []
    patternPairLst = [[TextProcessConfig.RePattern.NOFORMATREGION, TextProcessConfig.RePattern.NOFORMATTAG],
                      [TextProcessConfig.RePattern.CODEREGION, TextProcessConfig.RePattern.CODETAG]]
    for patternPair in patternPairLst:
        for match in re.finditer(patternPair[0], s, flags=re.DOTALL):
            tmp = s[match.start(): match.end()]
            tag = re.search(patternPair[1], tmp, flags=re.DOTALL)
            tagLength = tag.end() - tag.start()
            if tag.group(0).startswith('{code'):
                codeRegion.append([match.start(), match.end(), tag.group(0), tmp[tagLength: -6]])
            else:
                codeRegion.append([match.start(), match.end(), tag.group(0), tmp[tagLength: -tagLength]])
    return codeRegion


def minimalRegions(regions):
    """
    De duplication according to the start position and end position. Similar to makeMinimalSet(codeRegions) in
    codeFilter.py, but tags are not considered here. The results of this step are only used as intermediate products  to
    facilitate data cleaning.
    :param regions: list, list of all non-language regions mentioned above, each region contains 4 attributes, namely
             start index, end index, tag, non-language content
    :return: newRegions: list, list of all non-language regions mentioned above, each region contains 4 attributes, namely
               start index, end index, tag, non-language content
    """
    regions.sort(key=lambda x: x[0])
    newRegions = []
    for i in range(0, len(regions)):
        thisRegion = regions[i]
        isContained = False
        for j in range(0, len(newRegions)):
            thatRegion = newRegions[j]
            if thisRegion[1] <= thatRegion[1]:
                isContained = True
                break
            elif thisRegion[0] <= thatRegion[1]:
                thatRegion[3] = thatRegion[3] + thisRegion[3][(thatRegion[1] - thisRegion[0]):]
                thatRegion[1] = thisRegion[1]
                thatRegion[2] = thatRegion[2] + ',' + thisRegion[2]
                isContained = True
                newRegions[j] = thatRegion
                break
        if not isContained:
            newRegions.append([thisRegion[0], thisRegion[1], thisRegion[2], thisRegion[3]])
    newRegions.sort(key=lambda x: x[0], reverse=True)
    return newRegions


def cleanStr(s):
    """
    Remove all nonverbal content from the string
    :param s: string, raw
    :return: result: list, generate cleaned string for raw text, each result contains 3 columns, namely raw_string,
               non_language_regions, cleaned_string

    """
    if s:
        coloredRegion = getColoredRegion(s)
        s = removeColorTag(s, coloredRegion)
        resultLst = [s]
        stackTrace, cause = stackTraceFilter.javaStackTraceFilter(s)
        nonLanguageRegion = getNonLanguage(s)
        resultLst.append(nonLanguageRegion)
        miniRegion = minimalRegions(nonLanguageRegion)
        for r in miniRegion:
            s = s[: r[0]] + ' ' + s[r[1]:]
        for a in nonLanguageRegion:
            if a[2] == 'singlecomment':
                s = s + '\n' + a[3].strip()[2:]
            elif a[2] == 'multicomment':
                s = s + '\n' + a[3].strip()[2: -2]
        s = s.replace(u'\xa0', ' ')
        resultLst.append(s)
    else:
        resultLst = ['', [], '']
    return resultLst


def getNonLanguage(s):
    """
    Get all non-language region in string. Including getUrls(s), getExplicitCodeRegion(s), javaStackTraceFilter(s),
    javaCodeFilter(s)
    :param s: string
    :return: nonLanguage: list, list of non-language regions, each region contains 4 attributes, namely
               start index, end index, tag, non-language content
    """
    nonLanguage = []
    if s:
        nonLanguage.extend(getUrls(s))
        explicitCodeRegion = getExplicitCodeRegion(s)
        nonLanguage.extend(explicitCodeRegion)
        if len(explicitCodeRegion) == 0:
            stackTrace, cause = stackTraceFilter.javaStackTraceFilter(s)
            code = codeFilter.javaCodeFilter(s)
            nonLanguage.extend(stackTrace)
            nonLanguage.extend(cause)
            nonLanguage.extend(code)
        else:
            for region in explicitCodeRegion:
                stackTrace, cause = stackTraceFilter.javaStackTraceFilter(region[3])
                code = codeFilter.javaCodeFilter(region[3])
                for st in stackTrace:
                    st[0] = st[0] + region[0] + len(region[2])
                    st[1] = st[1] + region[0] + len(region[2])
                for ca in cause:
                    ca[0] = ca[0] + region[0] + len(region[2])
                    ca[1] = ca[1] + region[0] + len(region[2])
                for co in code:
                    co[0] = co[0] + region[0] + len(region[2])
                    co[1] = co[1] + region[0] + len(region[2])
                nonLanguage.extend(stackTrace)
                nonLanguage.extend(cause)
                nonLanguage.extend(code)
            region = explicitCodeRegion[0]
            newText = s[: region[0]]
            stackTrace, cause = stackTraceFilter.javaStackTraceFilter(newText)
            code = codeFilter.javaCodeFilter(newText)
            nonLanguage.extend(stackTrace)
            nonLanguage.extend(cause)
            nonLanguage.extend(code)
            for i in range(0, len(explicitCodeRegion)):
                region = explicitCodeRegion[i]
                if i == len(explicitCodeRegion) - 1:
                    newText = s[region[1]:]
                else:
                    newText = s[region[1]: explicitCodeRegion[i + 1][0]]
                stackTrace, cause = stackTraceFilter.javaStackTraceFilter(newText)
                code = codeFilter.javaCodeFilter(newText)
                for st in stackTrace:
                    st[0] = st[0] + region[1]
                    st[1] = st[1] + region[1]
                for ca in cause:
                    ca[0] = ca[0] + region[1]
                    ca[1] = ca[1] + region[1]
                for co in code:
                    co[0] = co[0] + region[1]
                    co[1] = co[1] + region[1]
                nonLanguage.extend(stackTrace)
                nonLanguage.extend(cause)
                nonLanguage.extend(code)
    nonLanguageFinal = []
    for n in nonLanguage:
        if n[2] == 'singlecomment':
            valid = True
            for m in nonLanguage:
                if m[2] == 'url' and m[0] < n[0] < m[1]:
                    valid = False
                    break
            if valid:
                nonLanguageFinal.append(n)
        else:
            nonLanguageFinal.append(n)
    return nonLanguageFinal


def removeUrlAndStack(s):
    if s:
        coloredRegion = getColoredRegion(s)
        s = removeColorTag(s, coloredRegion)
        nonLanguage = []
        stackTrace, cause = stackTraceFilter.javaStackTraceFilter(s)
        nonLanguage.extend(stackTrace)
        nonLanguage.extend(cause)
        nonLanguageFinal = []
        for n in nonLanguage:
            if n[2] == 'singlecomment':
                valid = True
                for m in nonLanguage:
                    if m[2] == 'url' and m[0] < n[0] < m[1]:
                        valid = False
                        break
                if valid:
                    nonLanguageFinal.append(n)
            else:
                nonLanguageFinal.append(n)

        miniRegion = minimalRegions(nonLanguageFinal)
        for r in miniRegion:
            s = s[: r[0]] + ' ' + s[r[1]:]
        for a in nonLanguageFinal:
            if a[2] == 'singlecomment':
                s = s + '\n' + a[3].strip()[2:]
            elif a[2] == 'multicomment':
                s = s + '\n' + a[3].strip()[2: -2]
        s = s.replace(u'\xa0', ' ')
    return s


def tokenize(s, level):
    """
    Tokenize a sentence or paragraph into words
    :param s: string
    :param level: string, {sentence, paragraph},indicate whether the input is a sentence or a paragraph
    :return: wordsLst: list, if the input is a sentence, it's a one-dimensional array, while if the input is a
               paragraph, it's a two-dimensional array.
    """
    pattern = TextProcessConfig.RePattern.TOKENIZPATTERN
    wordsLst = []
    if len(s) == 0:
        return wordsLst
    if level == 'sentence':
        wordsLst = [w.lower() for w in regexp_tokenize(s, pattern)]
    elif level == 'paragraph':
        sentLst = sent_tokenize(s)
        for sen in sentLst:
            wordsLst.append([w.lower() for w in regexp_tokenize(sen, pattern)])
    return wordsLst


@unique
class PartOfSpeech(Enum):
    J = wordnet.ADJ
    V = wordnet.VERB
    N = wordnet.NOUN
    R = wordnet.ADV


def reduction(wnl, wordsLst):
    """
    Restoring the part of speech of a word in a sentence
    :param wnl: WordNetLemmatizer
    :param wordsLst: list, a list of words
    :return: reducedWords: list
    """
    reducedWords = [wnl.lemmatize(word) for word in wordsLst]
    return reducedWords


def removeStopwords(wordsLst, stopwordsSet):
    """
    Remove stop words from sentences
    :param stopwordsSet: list, a list of stopwords
    :param wordsLst: list, a list of words
    :return: newWordsLst: list
    """
    newWordsLst = [w for w in wordsLst if w not in stopwordsSet and not isNumber(w)]
    return newWordsLst


def isNumber(s):
    """
    Judge whether the string is a number
    :param s: string
    :return: bool
    """
    if s.count(".") == 1:
        if s[0] == "-":
            s = s[1:]
        if s[0] == ".":
            return False
        s = s.replace(".", "")
        if s.isdigit():
            return True
        else:
            return False
    elif s.count(".") == 0:
        if s[0] == "-":
            s = s[1:]
        if s.isdigit():
            return True
        else:
            return False
    else:
        return False


def splitCamelWordAndConvertLower(word):
    """
    Used for processing camel naming method
    :param word: str, origin word
    :return: word_list: list
    """
    word_list = []
    cur = ''
    for c in word:
        if 'A' <= c <= 'Z':
            if len(cur) > 0:
                word_list.append(cur.lower())
            cur = ''
        cur += c
    if len(cur) > 0:
        word_list.append(cur.lower())
    return word_list


def removeDigits(text):
    """
    To remove digits from a string.
    :param text: str: origin text
    :return:
    """
    return re.sub(r'\d+', ' ', text)


def removeSpecialPatterns(pattern, text):
    """
    To remove special patterns from a string.
    :param pattern: re.pattern
    :param text: str: origin text
    :return:
    """
    return re.sub(pattern, ' ', text)


def removePunctuation(text):
    """
    To remove punctuation from a string.
    :param text: str: origin text
    :return:
    """
    return re.sub(r'[^\w\s]', ' ', text)