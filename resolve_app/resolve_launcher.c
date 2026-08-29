#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static const char *log_path = "/tmp/voice-enh-resolve.log";
static const char *python_path = "/Users/jenya/IdeaProjects/2026-2/voice-enh/.venv/bin/python";

int main(int argc, char **argv) {
    const char *input = NULL;
    for (int i = 1; i < argc; i++) {
        struct stat st;
        if (stat(argv[i], &st) == 0 && S_ISREG(st.st_mode)) {
            input = argv[i];
            break;
        }
    }

    FILE *log = fopen(log_path, "a");
    if (log) {
        fprintf(log, "launcher invoked argc=%d input=%s\n", argc, input ? input : "");
        fflush(log);
    }
    if (!input) {
        if (log) fclose(log);
        return 2;
    }

    char *const child_argv[] = {
        (char *)python_path, (char *)"-m", (char *)"voice_enh",
        (char *)"--resolve", (char *)input, NULL
    };
    if (log) {
        int fd = fileno(log);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        fclose(log);
    }
    execv(python_path, child_argv);
    return 127;
}
