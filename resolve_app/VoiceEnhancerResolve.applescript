on open droppedItems
    repeat with droppedItem in droppedItems
        set inputPath to POSIX path of droppedItem
        do shell script "printf '%s\\n' " & quoted form of ((current date) as text) & " event input: " & quoted form of inputPath & " >> /tmp/voice-enh-resolve.log"
        do shell script "/Users/jenya/IdeaProjects/2026-2/voice-enh/bin/voice-enh-resolve " & quoted form of inputPath
    end repeat
end open

on run
    try
        set clipboardPath to the clipboard as text
        if clipboardPath is not "" then
            do shell script "/Users/jenya/IdeaProjects/2026-2/voice-enh/bin/voice-enh-resolve " & quoted form of clipboardPath
        end if
    on error errorMessage
        do shell script "printf '%s\\n' " & quoted form of ((current date) as text) & " run error: " & quoted form of errorMessage & " >> /tmp/voice-enh-resolve.log"
    end try
end run

on idle
    return 1
end idle
