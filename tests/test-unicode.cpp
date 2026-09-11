#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

static int check(const char * name, const std::string & text,
                 const std::vector<std::string> & regex_exprs,
                 const std::vector<std::string> & expected) {
    const auto actual = unicode_regex_split(text, regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "%s: unexpected split:", name);
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    return 0;
}

int main() {
    int n_fail = 0;

    n_fail += check("simple", " ~foo",
        { "[~][A-Za-z]+| ?[\\p{S}]+|\\s+" },
        { " ~", "foo" });

    // K2-Horizon letter runs take marks and ZWNJ/ZWJ, so a ZWNJ must not end a run.
    // "A" + "mi" + U+200C + "khaham" + " " + "1"
    const std::string k2_text = "Aمی‌خواهم 1";

    n_fail += check("k2-horizon zwnj", k2_text,
        { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" },
        { "Aمی‌خواهم", " ", "1" });

    return n_fail;
}
