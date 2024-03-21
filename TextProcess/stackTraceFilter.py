import re

from TextProcess import TextProcessConfig


def findJavaExceptions(s):
    """
    Exception is the base of stacktrace and cause. Find all JAVA exceptions in the string based on the regular expression
    :param s: string
    :return: exceptionLst: list, list of exception regions, each cause region contains 4 attributes, namely start index,
               end index, tag, exception
    """
    exceptionLst = []
    pattern = TextProcessConfig.StackTracePattern.JAVAEXCEPTION
    for match in re.finditer(pattern, s, flags=re.DOTALL | re.MULTILINE):
        if len(exceptionLst) > 0:
            gapStr = s[exceptionLst[-1][0]: match.start()]
            if gapStr.count('at') >= 10:
                exceptionLst.append([match.start(), match.end(), 'exception', s[match.start(): match.end()]])
        else:
            exceptionLst.append([match.start(), match.end(), 'exception', s[match.start(): match.end()]])
    return exceptionLst


def findJavaStackTrace(s):
    """
    Find all JAVA stacktrace and cause in the string based on the regular expression
    :param s: string
    :return: stackTraceLst: list, list of stacktrace regions, each stacktrace region contains 4 attributes, namely
               start index, end index, tag, stacktrace
             causeLst: list, list of cause regions, each cause region contains 4 attributes, namely start index,
               end index, tag, cause
    """
    stackTraceLst = []
    causeLst = []
    pattern = TextProcessConfig.StackTracePattern.JAVASTACKTRACE
    for match in re.finditer(pattern, s, flags=re.DOTALL | re.MULTILINE):
        stackTraceLst.append([match.start(), match.end(), 'stacktrace', s[match.start(): match.end()]])
    pattern = TextProcessConfig.StackTracePattern.JAVACAUSE
    for match in re.finditer(pattern, s, flags=re.DOTALL | re.MULTILINE):
        causeLst.append([match.start(), match.end(), 'cause', s[match.start(): match.end()]])
    return stackTraceLst, causeLst


def javaStackTraceFilter(s):
    """
    Public. Find all JAVA stacktrace and cause in the string based on the regular expression
    :param s: string
    :return: stackTraceLst: list, list of stacktrace regions, each stacktrace region contains 4 attributes, namely
               start index, end index, tag, stacktrace
             causeLst: list, list of cause regions, each cause region contains 4 attributes, namely start index,
               end index, tag, cause
    """
    stackTraceLst = []
    causeLst = []
    exceptionLst = findJavaExceptions(s)
    if len(exceptionLst) > 0:
        for i in range(0, len(exceptionLst) - 1):
            region = s[exceptionLst[i][0]: exceptionLst[i + 1][0]]
            stackTraceTmp, causeTmp = findJavaStackTrace(region)
            for st in stackTraceTmp:
                st[0] = st[0] + exceptionLst[i][0]
                st[1] = st[1] + exceptionLst[i][0]
            for c in causeTmp:
                c[0] = c[0] + exceptionLst[i][0]
                c[1] = c[1] + exceptionLst[i][0]
            stackTraceLst.extend(stackTraceTmp)
            causeLst.extend(causeTmp)
            if len(stackTraceTmp) == 0 and len(causeTmp) == 0:
                stackTraceLst.append(exceptionLst[i])
        region = s[exceptionLst[-1][0]:]
        stackTraceTmp, causeTmp = findJavaStackTrace(region)
        for st in stackTraceTmp:
            st[0] = st[0] + exceptionLst[-1][0]
            st[1] = st[1] + exceptionLst[-1][0]
        for c in causeTmp:
            c[0] = c[0] + exceptionLst[-1][0]
            c[1] = c[1] + exceptionLst[-1][0]
        stackTraceLst.extend(stackTraceTmp)
        causeLst.extend(causeTmp)
        if len(stackTraceTmp) == 0 and len(causeTmp) == 0:
            stackTraceLst.append(exceptionLst[-1])
    return stackTraceLst, causeLst
