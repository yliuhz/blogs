## Math with [KaTex](https://khan.github.io/KaTeX/function-support.html) ##

More in this [Hugo HP](https://gohugo.io/content-management/formats/#mathjax-with-hugo).

#### 1. `_{` problem

Replace `_{` by `\_{` to disable Markdown transform.
Such as:  
```
s_t = f(U x_t + W s\_{t-1} )
```

#### 2. `\` problem

Replace `\` by `\\` to disable Markdown transform.
Such as:  
```
\\{ A\_{i,i} \\}
```

#### 3. Aligned layout with empty before `=`
Prepend `\` before `&=`.
Such as:
```
\begin{aligned}
E_t(y_t, \hat{h_y}) &= -y_t log{\hat{y_t}} \\cr
E(y, \hat{y})       &= \sum_t{E_t(y_t, \hat{h_y})} \\cr
                  \ &= -\sum_t{y_t log{h_y})}
\end{aligned}
```

## Useful Links
* [Math Dict](http://www.tudientoan.com/)
* [KaTex Functions](https://khan.github.io/KaTeX/function-support.html)
* [Unicode Emoji](https://unicode.org/emoji/charts/text-style.html)