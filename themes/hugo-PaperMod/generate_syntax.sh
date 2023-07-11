# generate_syntax.sh
#!/usr/bin/env bash
set -euo pipefail

# the cp is just a way to track which css you might end up liking
# cp assets/css/extended/syntax.css assets/css/extended/syntax_$(date +'%Y_%m_%d_%H_%M_%S').css
rm assets/css/extended/syntax*.css
hugo gen chromastyles --style=$1 > assets/css/extended/syntax_light.css
hugo gen chromastyles --style=$2 > assets/css/extended/syntax_dark.css

cat assets/css/extended/syntax_light.css > assets/css/extended/syntax.css
echo "@media screen and (prefers-color-scheme: dark) {" >> assets/css/extended/syntax.css
cat assets/css/extended/syntax_dark.css >> assets/css/extended/syntax.css
echo "}" >> assets/css/extended/syntax.css
