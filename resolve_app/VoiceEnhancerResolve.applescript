on open droppedItems
    repeat with droppedItem in droppedItems
        set inputPath to POSIX path of droppedItem
        do shell script "/Users/jenya/IdeaProjects/2026-2/voice-enh/bin/voice-enh-resolve " & quoted form of inputPath
    end repeat
end open

on run
end run

on idle
    return 1
end idle
