%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pnas-new.cls, v1.47, 2025/11/01
%
% This class file enables authors to prepare research
% articles for submission to PNAS.
%
% Please note that whilst this template provides a
% preview of the typeset manuscript for submission, it
% will not necessarily be the final publication layout.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either
% version 1.3 of this license or any later version.
% The latest version of this license is in
% http://www.latex-project.org/lppl.txt and
% version 1.3 or later is part of all distributions
% of LaTeX version 2005/12/01 or later.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% For use with latex+dvipdfm, pdflatex, xelatex & lualatex
% For compiling with plain latex, please use latex+dvipdfm
% to produce the PDF, not dvis -> ps -> pdf nor dvipdf
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pnas-new}[2025/11/01, v1.47]
\AtEndOfClass{\RequirePackage{microtype}}
% Option for line numbers
%\newif\if@pnaslineno
%\DeclareOption{lineno}{\@pnaslinenotrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions*
\LoadClass{extarticle}

\widowpenalty=10000 % To avoid widow & orphan lines
\clubpenalty=10000  % To avoid single line remain at preceding page
\hyphenpenalty=750 % For proper hyphenation

%% Fonts and language
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{lmodern}
\RequirePackage[scaled]{helvet}
%\RequirePackage[T1]{fontenc}
\RequirePackage{lettrine} % For dropped capitals

%% For the Significance Statement & footnote on the first page
\RequirePackage{afterpage}
\RequirePackage{ifpdf,ifxetex}
\ifpdf\else
  \ifxetex\else
    \def\pgfsysdriver{pgfsys-dvipdfm.def}
    \pdfpagewidth=\paperwidth
    \pdfpageheight=\paperheight
\fi\fi
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage[framemethod=tikz]{mdframed}

%% Hyperlinking
\RequirePackage[colorlinks=true, allcolors=blue]{hyperref}

%% Set up main title page fonts
\newcommand{\headerfont}{\normalfont\sffamily\fontsize{7}{9} \selectfont}
\newcommand{\footerfont}{\normalfont\sffamily\fontsize{7}{9} \selectfont}
\newcommand{\titlefont}{\fontfamily{lmss}\bfseries\fontsize{15}{20}\selectfont}
\newcommand{\dropcapfont}{\fontfamily{lmss}\bfseries\fontsize{26pt}{28pt}\selectfont}
\newcommand{\datesfont}{\normalfont\sffamily\fontsize{7}{8}\selectfont}
\newcommand{\absfont}{\normalfont\bfseries\fontsize{10}{12}\selectfont}
\newcommand{\keywordsfont}{\normalfont\sffamily\fontsize{7}{9}\selectfont}
\newcommand{\copyrightfont}{\normalfont\rmfamily\fontsize{6}{8}\selectfont}

%% Set URL link color & font
\renewcommand\UrlFont{\color{black}\sffamily}

%% Author and affiliation
\RequirePackage{authblk}
\setlength{\affilsep}{8.5pt} % 16.5pts between base of author line and base of affil line
\renewcommand\Authfont{\color{color0}\sffamily\fontsize{6}{7.5}\selectfont}
\renewcommand\Affilfont{\color{color0}\sffamily\fontsize{6}{7.5}\selectfont}

\renewcommand\AB@affilsepx{; \protect\Affilfont}

\renewcommand\Authands{, and }

\renewcommand\@author{\ifx\AB@affillist\AB@empty\AB@author\else
      \ifnum\value{affil}>\value{Maxaffil}\def\rlap##1{##1}%
    \AB@authlist%\\[\affilsep]\AB@affillist
    \else  \AB@authors\fi\fi}

%% Choose template type
\newcommand*{\templatetype}[1]{%
  \RequirePackage{#1}}

%% Options for element switching
\RequirePackage{xifthen}
\newboolean{shortarticle}
\newboolean{singlecolumn}

%% For numbering just one line of an equation
\newcommand\numberthis{\addtocounter{equation}{1}\tag{\theequation}}

%% Watermark
\newboolean{displaywatermark}
\setboolean{displaywatermark}{false} % Set to false to remove the watermark
\AtBeginDocument{%
  \ifthenelse{\boolean{displaywatermark}}{%
  \RequirePackage{draftwatermark}
  \SetWatermarkAngle{45}
  \SetWatermarkColor{gray!20}
  \SetWatermarkFontSize{3cm}
  \SetWatermarkText{{\fontfamily{bch}\bfseries DRAFT}}
}{}
}

%% Copyright statement (not used)
\newboolean{displaycopyright}
\setboolean{displaycopyright}{false} % Confirmed as not required
\RequirePackage{textcomp} % For copyright symbol styling
\newcommand{\copyrightstatement}{\, \textcopyright\, 2015 by The National Academy of Sciences of the USA}

%% Graphics, tables and other formatting
\RequirePackage{graphicx,xcolor}
\RequirePackage{colortbl}
\RequirePackage{booktabs}
\RequirePackage{algorithm}
\RequirePackage[noend]{algpseudocode}
\RequirePackage{changepage}
\RequirePackage[twoside,%
				letterpaper,includeheadfoot,%
				layoutsize={8.125in,10.875in},%
                layouthoffset=0.1875in,%
                layoutvoffset=0.0625in,%
                %left=22.5pt,%
                %right=48pt,%
                top=43pt,% 10pt provided by headsep
                bottom=44pt,%
                headheight=0pt,% No Header
                headsep=-2pt,%
                footskip=25pt,
                marginparwidth=38pt]{geometry}
\RequirePackage[labelfont={bf,sf},%
                labelsep=period,%
                figurename=Fig.]{caption}
\setlength{\columnsep}{13.5pt} % Distance between the two columns of text
\setlength{\parindent}{12pt} % Paragraph indent
\setlength{\headheight}{12pt}
%% Set document color scheme
\definecolor{black50}{gray}{0.5} % 50% black for hrules
\definecolor{color0}{RGB}{0,0,0} % Base
\definecolor{color1}{RGB}{59,90,198} % author email, doi
\definecolor{color2}{RGB}{16,131,16} %
% For sig statement box
\definecolor{pnasbluetext}{RGB}{0,101,165} %
%\definecolor{pnasblueback}{RGB}{205,217,235} %
\definecolor{pnasblueback}{cmyk}{.07,.02,.01,0}%
\definecolor{pantone}{cmyk}{1,.46,.02,.002}%used for significant head
%\definecolor{pnasbluetext}{RGB}{0,115,209} % Not used
%\definecolor{pnasblueback}{RGB}{210,230,247} % Not used

%%%%%%%%%%%%Straive changes 2025%%%%%%%%%%%%%%
\advance\paperwidth by -.375in%
\advance\paperheight by -.13in
\setlength{\topmargin}{-50pt}
\setlength{\columnwidth}{247pt}%
\setlength{\columnsep}{18pt}%
\setlength{\evensidemargin}{-35.1pt}%
\setlength{\oddsidemargin}{-35.1pt}%
\setlength{\textwidth}{512pt}%
\advance\textheight by2.37\baselineskip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Bibliography
%% v1.41 (15/11/2016)
%% Use pnas-new.bst if available
%% If not, try to use pnas2011.bst instead
%% If both aren't available, fall back to
%% unsrtnat
\RequirePackage[numbers,sort&compress,merge,round]{natbib}
\setlength{\bibsep}{0.0pt}
\IfFileExists{pnas-new.bst}{
  \bibliographystyle{pnas-new}
}{
  \IfFileExists{pnas2011.bst}{
    \bibliographystyle{pnas2011}
    \PackageWarning{pnas-new}{pnas-new.bst not found; using pnas2011.bst instead}
  }{
    \bibliographystyle{unsrtnat}
    \PackageWarning{pnas-new}{pnas-new.bst and pnas2011.bst not found; falling back to unsrtnat instead}
  }
}
\renewcommand{\bibsection}{} % Remove header
\renewcommand\bibfont{\normalfont\sffamily\fontsize{6}{8}\selectfont} % set font to be sans serif
%% April 29, 2019: abbreviate journal names
\RequirePackage{jabbrv}

\renewcommand\@biblabel[1]{ #1.} % Remove brackets from label
\def\tagform@#1{\maketag@@@{\bfseries(\ignorespaces#1\unskip\@@italiccorr)}}
\renewcommand{\eqref}[1]{\textup{{\normalfont Eq.~(\ref{#1}}\normalfont)}}

\renewenvironment{thebibliography}[1]{%
\bibsection
 \parindent\z@
 \labelsep2.8pt
 \bibpreamble
\fontsize{6.75pt}{8pt}\selectfont
 \list{\@biblabel{\the\c@NAT@ctr}}{\@bibsetup{#1}\global\c@NAT@ctr\z@}%
 \ifNAT@openbib
   \renewcommand\newblock{\par}%
 \else
   \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
 \fi
 \sloppy
 \clubpenalty4000\widowpenalty4000
 \sfcode`\.\@m
 \let\NAT@bibitem@first@sw\@firstoftwo
    \let\citeN\cite \let\shortcite\cite
    %\let\citeasnoun\cite\xbibfont
	\bibfont\raggedright
}{%
 \bibitem@fin
 \bibpostamble
 \def\@noitemerr{%
  \PackageWarning{natbib}{Empty `thebibliography' environment}%
 }%
 \endlist
 \bibcleanup
 \label{LastPage}
}%


%% Figure caption style
\DeclareCaptionFormat{pnasformat}{\normalfont\sffamily\fontsize{7}{9}\selectfont#1#2#3}
\captionsetup*{format=pnasformat}

%% Table style
\RequirePackage{etoolbox}
\captionsetup*[table]{labelfont+={small},textfont+={small,sf,bf},skip=10pt,position=above}
% booktabs provide nice spacing, but rule widths and distances need fixing
\setlength{\heavyrulewidth}{0.5pt}
\setlength{\lightrulewidth}{0.5pt}
\setlength{\aboverulesep}{1.5pt}
\setlength{\belowrulesep}{1.5pt}
\setlength{\belowbottomsep}{10pt}
\AtBeginEnvironment{tabular}{
\sffamily\fontsize{7.5}{10}\selectfont
}
\AtBeginEnvironment{tabular*}{
\sffamily\fontsize{7.5}{10}\selectfont
}
\newcommand{\addtabletext}[1]{{\setlength{\leftskip}{9pt}\fontsize{7}{9}\selectfont#1}}



%% Equation numbering - use square brackets

\renewcommand\tagform@[1]{\maketag@@@ {\bfseries[\ignorespaces #1\unskip \@@italiccorr ]}}


%% Headers and footers
\RequirePackage{fancyhdr}  % custom headers/footers
\RequirePackage{lastpage}  % Number of pages in the document
\pagestyle{fancy}          % Enables the custom headers/footers

\fancypagestyle{firststyle}{
   \fancyfoot[L]{\footerfont \textbf{PNAS}\hspace{7pt}\today\hspace{7pt}LaTex v2025\hspace{7pt}Vol. XXX\hspace{7pt}No. XX\hspace{7pt}eXXXXXXXXXX}
   \fancyfoot[R]{\footerfont\@ifundefined{@doi}{}{\@doi}\hspace{7pt}\textbf{\thepage of \pageref*{LastPage}}}

\newsavebox{\PNAS@linecount@bx}
\newlength\PNAS@linecount@bxht
\newcount\PNAS@linecount
\PNAS@linecount\@ne\relax
\def\PNAS@mk@linecount{%
    \savebox{\PNAS@linecount@bx}[4em][t]{\parbox[t]{4em}{\raggedleft\vskip4pc\sffamily
        \normalsize
        \setlength{\PNAS@linecount@bxht}{0pt}%
        \loop{\raggedleft\scriptsize\the\PNAS@linecount}\\
        \global\advance\PNAS@linecount by \@ne
        \addtolength{\PNAS@linecount@bxht}{\baselineskip}%
        \ifdim\PNAS@linecount@bxht<0.93\textheight\repeat
        {\raggedleft\scriptsize\the\PNAS@linecount}\hfill
        \global\advance\PNAS@linecount by \@ne}}}

\def\PNAS@linecountLO{%
  \PNAS@mk@linecount
  \begin{picture}(0,0)%
    \put(-50,7){\usebox{\PNAS@linecount@bx}}%
  \end{picture}%
}
\def\PNAS@linecountRO{%
  \PNAS@mk@linecount
  \begin{picture}(0,0)%
    \put(-13,7){\usebox{\PNAS@linecount@bx}}%
  \end{picture}%
}
\def\PNAS@linecountLE{%
  \PNAS@mk@linecount
  \begin{picture}(0,0)%
    \put(-28,4){\usebox{\PNAS@linecount@bx}}%
  \end{picture}%
}
\def\PNAS@linecountRE{%
  \PNAS@mk@linecount
  \begin{picture}(0,0)%
    \put(11,4){\usebox{\PNAS@linecount@bx}}%
  \end{picture}%
}

\fancyhead[LO]{\PNAS@linecountLO}
\fancyhead[RO]{\PNAS@linecountRO}
\fancyhead[LE]{\PNAS@linecountLE}
\fancyhead[RE]{\PNAS@linecountRE}
}

\newsavebox{\PNAS@linecount@bxsecpage}
\newlength\PNAS@linecount@bxhtsecpage
%\newcount\PNAS@linecount
%\PNAS@linecount\@ne\relax
\def\PNAS@mk@linecountsecpage{%
    \savebox{\PNAS@linecount@bxsecpage}[4em][t]{\parbox[t]{4em}{\raggedleft\vskip15pt\sffamily
        \normalsize
        \setlength{\PNAS@linecount@bxhtsecpage}{0pt}%
        \loop{\raggedright\scriptsize\the\PNAS@linecount}\\
        \global\advance\PNAS@linecount by \@ne
        \addtolength{\PNAS@linecount@bxhtsecpage}{\baselineskip}%
        \ifdim\PNAS@linecount@bxhtsecpage<0.98\textheight\repeat
        {\raggedright\scriptsize\the\PNAS@linecount}\hfill
        \global\advance\PNAS@linecount by \@ne}}}

\def\PNAS@linecountsecpageLO{%
  \PNAS@mk@linecountsecpage
  \begin{picture}(0,0)%
    \put(-50,7){\usebox{\PNAS@linecount@bxsecpage}}%
  \end{picture}%
}
\def\PNAS@linecountsecpageRO{%
  \PNAS@mk@linecountsecpage
  \begin{picture}(0,0)%
    \put(-13,7){\usebox{\PNAS@linecount@bxsecpage}}%
  \end{picture}%
}

\def\PNAS@linecountsecpageLE{%
  \PNAS@mk@linecountsecpage
  \begin{picture}(0,0)%
    \put(-50,7){\usebox{\PNAS@linecount@bxsecpage}}%
  \end{picture}%
}
\def\PNAS@linecountsecpageRE{%
  \PNAS@mk@linecountsecpage
  \begin{picture}(0,0)%
    \put(-13,7){\usebox{\PNAS@linecount@bxsecpage}}%
  \end{picture}%
}

\fancyhead[LO]{\PNAS@linecountsecpageLO}
\fancyhead[RO]{\PNAS@linecountsecpageRO}
\fancyhead[LE]{\PNAS@linecountsecpageLE}
\fancyhead[RE]{\PNAS@linecountsecpageRE}
% Headers
%\fancyhead[LE,RO]{}
%\fancyhead[LO,RE]{}
% Footers
\lfoot{}%
\cfoot{}%
\rfoot{}%

\fancyfoot[LE]{\footerfont\textbf{\thepage} \textbf{of} \textbf{\pageref*{LastPage}}\hspace{7pt}\@ifundefined{@doi}{}{\@doi}}
\fancyfoot[RO]{\footerfont\@ifundefined{@doi}{}{\@doi}\hspace{7pt}\textbf{\thepage} \textbf{of} \textbf{\pageref*{LastPage}}}
%\fancyfoot[RE,LO]{\footerfont\@ifundefined{@leadauthor}{}{\@leadauthor}\ifnum \value{authors} > 1\hspace{5pt}\textit{et al.}\fi}
\fancyfoot[RE]{\footerfont pnas.org}
\fancyfoot[LO]{\footerfont \textbf{PNAS}\hspace{7pt}\today\hspace{7pt}LaTex v2025 \hspace{7pt}Vol. XXX\hspace{7pt}No. XX\hspace{7pt} eXXXXXXXXXX}

\renewcommand{\headrulewidth}{0pt}% % No header rule
\renewcommand{\footrulewidth}{0pt}% % No footer rule

%% Section/subsection/paragraph set-up
\RequirePackage[explicit]{titlesec}
\setcounter{secnumdepth}{6}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
\renewcommand{\theparagraph}{\thesubsubsection.\arabic{paragraph}}

\titleformat{\section}
  {\large\sffamily\bfseries}
  {\thesection.}
  {0.5em}
  {#1}
  []
\titleformat{name=\section,numberless}
  {\large\sffamily\bfseries}
  {}
  {0em}
  {#1}
  []
\titleformat{\subsection}[runin]
  {\sffamily\bfseries\fontsize{8}{10}\selectfont}
  {\thesubsection.}
  {0.5em}
  {#1. }
  []
\titleformat{\subsubsection}[runin]
  {\sffamily\small\bfseries\itshape}
  {\thesubsubsection.}
  {0.5em}
  {#1. }
  []
\titleformat{\paragraph}[runin]
  {\sffamily\small\itshape\bfseries}
  {\indent\theparagraph. }
  {0em}
  {#1.}
\titlespacing*{\section}{0pc}{3ex \@plus4pt \@minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex \@plus3pt \@minus2pt}{6pt}
\titlespacing*{\subsubsection}{0pc}{2ex \@plus2.5pt \@minus1.5pt}{6pt}
\titlespacing*{\paragraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{6pt}

%% Article meta data additional fields
\newcommand{\additionalelement}[1]{\def\@additionalelement{#1}}
\newcommand{\dates}[1]{\def\@dates{#1}}
\newcommand{\doi}[1]{\def\@doi{#1}}
\newcommand{\leadauthor}[1]{\def\@leadauthor{#1}}
\newcommand{\etal}[1]{\def\@etal{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newcommand{\authorcontributions}[1]{\def\@authorcontributions{#1}}
\newcommand{\authordeclaration}[1]{\def\@authordeclaration{#1}}
\newcommand{\equalauthors}[1]{\def\@equalauthors{#1}}
\newcommand{\correspondingauthor}[1]{\def\@correspondingauthor{#1}}
\newcommand{\significancestatement}[1]{\def\@significancestatement{\raggedright\fontsize{8.5}{12}\selectfont#1}}
\newcommand{\matmethods}[1]{\def\@matmethods{#1}}
\newcommand{\acknow}[1]{\def\@acknow{#1}}


%% Dropped capital for first letter of main text
\newcommand{\dropcap}[1]{\lettrine[lines=2,lraise=0.05,findent=0.1em, nindent=0em]{{\dropcapfont{#1}}}{}}

%% Abstract formatting
\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract
\long\gdef\theabstract{\ignorespaces#1}
\def\go{\end{abstract}}\else
\typeout{^^J^^J PLEASE DO NOT USE ANY \string\begin\space \string\end^^J
COMMANDS WITHIN ABSTRACT^^J^^J}#1\end{#2}
\gdef\theabstract{\vskip12pt BADLY FORMED ABSTRACT: PLEASE DO
NOT USE {\tt\string\begin...\string\end} COMMANDS WITHIN
THE ABSTRACT\vskip12pt}\let\go\relax\fi
\go}

% Define an environment with abstract content and styling
\newcommand{\abscontent}{
\noindent
\parbox{\dimexpr1.35\linewidth}{%
    \vskip3pt%
	\absfont \theabstract
}%
\vskip10pt%
\noindent
\parbox{\dimexpr1.35\linewidth}{%
{
 \keywordsfont \@ifundefined{@keywords}{}{\@keywords}}%
}
\vskip12pt%
}

\newcommand{\secondpage}{\@ifstar{\pnaspar}{\parabreak}}

\newcommand{\Parasplit}{\parfillskip=0pt\eject\clearpage\setlength{\columnwidth}{\linewidth}\setlength{\columnwidth}{247pt}\setlength{\linewidth}{247pt}\setlength{\hsize}{247pt}\parfillskip=0pt plus 1fil\noindent}
\newcommand{\Endparasplit}{\vfill\eject\clearpage\setlength{\columnwidth}{\linewidth}\setlength{\columnwidth}{247pt}\setlength{\linewidth}{247pt}\setlength{\hsize}{247pt}\parfillskip=0pt plus 1fil}
\newcommand{\normalcolwidth}{\setlength{\columnwidth}{\linewidth}\setlength{\columnwidth}{247pt}\setlength{\linewidth}{247pt}\setlength{\hsize}{247pt}}
\newcounter{parcount}

\newcommand*\replicate[1]{%
  \expandafter\replicate@aux\romannumeral\number\numexpr #1\relax000Q{}
}
\newcommand*\replicate@aux[1]{\csname replicate@aux@#1\endcsname}
\newcommand\replicate@aux@m{}
\long\def\replicate@aux@m#1Q#2#3{\replicate@aux#1Q{#2#3}{#3}}
\newcommand\replicate@aux@Q[2]{#1}
\DeclareRobustCommand*\adjlinewidth[3]{%
    \parshape = #1
      \replicate{#1-1}{#2\textwidth #3\textwidth}%
      0pt\columnwidth
}


\newcommand{\Firstpage}{%
\setlength{\columnwidth}{0.695\textwidth}
\setlength{\linewidth}{0.66\textwidth}
\setlength{\hsize}{0.66\textwidth}
\setlength{\columnsep}{10pt}
\setlength{\parskip}{0pt}}

\newcommand{\firstpage}[2][1]{%
\let\FPeverypar\everypar
\newtoks\everypar
\newtoks\listeverypar
\setlength{\columnwidth}{0.7\textwidth}
\setlength{\linewidth}{0.655\textwidth}
\setlength{\hsize}{0.655\textwidth}
\setlength{\parskip}{0pt}\FPeverypar{\the\everypar\the\listeverypar\stepcounter{parcount}\ifnum\theparcount=0\relax\else\ifnum\theparcount>0\ifnum\theparcount=#2\normalcolwidth\adjlinewidth{#1}{0}{0.655}\fi\fi\fi}}


\newcounter{BalanceAtReference}
\setcounter{BalanceAtReference}{\z@}
\newcounter{ReferenceIndexForBalancing}

\def\@balancereference{%
  \ifnum\value{ReferenceIndexForBalancing}=\value{BalanceAtReference}
\ifnum\value{BalanceAtReference}=\z@
\relax
\else
    \vfill\eject
  \fi\fi
  \stepcounter{ReferenceIndexForBalancing}
}
\pretocmd{\bibitem}{\@balancereference}
  {} % on success
  {\@latex@error{Patching \bibitem failed}{\@ehd}}



\RequirePackage{cuted}

\def\@outputdblcol{%
  \if@firstcolumn
    \global \@firstcolumnfalse
    \global \setbox\@leftcolumn \box\@outputbox
    \splitmaxdepth\maxdimen
        \vbadness\maxdimen
        \setbox\@outputbox\vbox{\unvbox\@outputbox\unskip}%
        \setbox\@outputbox\vsplit\@outputbox to\maxdimen
  \else
\ifnum\value{page}=1
%\linewidth=0.655\textwidth
%\hsize=0.655\textwidth
\global \@firstcolumntrue
\setbox\@outputbox \vbox {%
                         \hb@xt@\textwidth {%
                           \hb@xt@0.7\textwidth {%
                             \box\@leftcolumn \hss}%
                           \hfil
                           \vrule \@width\columnseprule
                           \hfil
                           \hb@xt@\columnwidth {%
                             \box\@outputbox \hss}%
                                             }%
                              }%
\else
\global \@firstcolumntrue
    \setbox\@outputbox \vbox {%
                         \hb@xt@\textwidth {%
                           \hb@xt@\columnwidth {%
                             \box\@leftcolumn \hss}%
                           \hfil
                           \vrule \@width\columnseprule
                           \hfil
                           \hb@xt@\columnwidth {%
                             \box\@outputbox \hss}%
                                             }%
                              }%
                              \fi
    \ifvoid\hold@viper
    \else
      \setbox\@outputbox \vbox{\box\hold@viper\box\@outputbox}%
    \fi
    \@combinedblfloats
    \@outputpage
    \begingroup
      \@dblfloatplacement
      \@startdblcolumn
      \@whilesw\if@fcolmade \fi
        {\@outputpage
         \@startdblcolumn}%
      \ifvoid\@viper
      \else
        \global\setbox\@viper\vbox{%
                \vskip-\stripsep\unvbox\@viper}\@viperoutput
      \fi
    \endgroup
  \fi
}

\newcommand\bibsplit[1][]{\begin{strip}\vspace*{-12pt}\end{strip}\setcounter{BalanceAtReference}{#1}}


% Option to format abstract differently for certain layouts (not used)
\newcommand{\abscontentformatted}{
\abscontent
}

%% Manual adjustment to line up main content with line numbers
\newlength\pnas@vertadjust
\newcommand\verticaladjustment[1]{\setlength{\pnas@vertadjust}{#1}}

\definecolor{typecol}{cmyk}{0,.67,.9,.0}%
\definecolor{gray80}{gray}{0.75}

\newcommand{\articletype}[1]{\def\@articletype{\footnotesize\sffamily#1}}

%% Custom title page
\renewcommand{\@maketitle}{\vskip-8.5\p@%
{\includegraphics[height=24.5pt,width=66.4pt]{PNAS_Logo.pdf}%
\put(28.9,9.05){\color{typecol}\sffamily\bfseries\footnotesize RESEARCH ARTICLE}\put(102,3.55){\color{gray80}{\rule{0.5pt}{16pt}}}\put(110,9.05){\@articletype}\par\vskip11\p@%
\ifthenelse{\boolean{shortarticle}}
  {\ifthenelse{\boolean{singlecolumn}}{}{
    {\raggedright\baselineskip= 24pt\titlefont \@title\par}%
    \vskip10pt% 21pts between base of title and base of author line
    {\raggedright \@author\par}
    \vskip8pt% 16pts between base of affiliations and base of dates line
    {\raggedright \datesfont \@ifundefined{@dates}{}{\@dates}\par}
    \vskip12pt%
    }}
  {% else
    %
    \vskip10pt%
    {\raggedright\baselineskip= 20pt\titlefont \@title\par}%
    \vskip10pt% 21pts between base of title and base of author line
    {\raggedright \@author\par}
    \vskip8pt% 16pts between base of affiliations and base of dates line
    {\raggedright \datesfont \@ifundefined{@dates}{}{\@dates}\par}
    \vskip12pt
    {%
    \abscontent
    }%
    \vskip25pt%
  }%
%%%
\@additionalelement
}%
\vskip\pnas@vertadjust
}

%%%% Adding line numbers
%\if@twocolumn
%  \RequirePackage[switch,mathlines]{lineno}
%\else
%  \RequirePackage[mathlines]{lineno}
%\fi
%
%\if@pnaslineno
%  \linenumbers
%
%  \patchcmd{\abscontent}{\noindent}{\noindent\nolinenumbers}{}{}
%  \patchcmd{\abscontent}{\theabstract}{\internallinenumbers\theabstract}{}{}
%  \appto{\abscontent}{\linenumbers*}
%
%  \if@twocolumn
%  \else
%    \preto{\@maketitle}{\nolinenumbers}
%  \fi
%\fi
%

%% Footnotes set up
\RequirePackage[flushmargin,ragged,symbol*]{footmisc}
\renewcommand*{\footnotelayout}{\normalfont\sffamily\fontsize{6}{8}\selectfont} % set the footnote font
\renewcommand{\footnoterule}{% Set the footnote hrule style
  \kern -3pt
  {\color{black50} \hrule width 72pt height 0.25pt}
  \kern 2.5pt
}

%% Set up the acknowledgments field
\titleclass{\acknow@section}{straight}[\part]
\newcounter{acknow@section}
\providecommand*{\toclevel@acknow@section}{0}
% Format is set for research articles by default
\titleformat{\acknow@section}[runin]
   {\sffamily\normalsize\bfseries}
   {}
   {0em}
   {#1.}
   []
\titlespacing{\acknow@section}
	{0pt}
	{3.25ex plus 1ex minus .2ex}
	{1.5ex plus .2ex}


\newcommand{\showacknow}{% Display acknowledgments section
\@ifundefined{@acknow}{}{
\vskip 3.25ex plus 1ex minus .2ex
\noindent{\sffamily\bfseries\fontsize{7}{9}\selectfont ACKNOWLEDGMENTS.\hspace{0.5ex plus .2ex}}
\sffamily\small\@acknow}
}

\newcommand{\dataavail}[1]{% Display acknowledgments section
\vskip 3.25ex plus 1ex minus .2ex
\noindent{\sffamily\bfseries\fontsize{7}{9}\selectfont Data, Materials, and Software Availability.\hspace{0.5ex plus .2ex}}
\sffamily\small#1}



%% Set up the materials&methods field
\titleclass{\matmethods@section}{straight}[\part]
\newcounter{matmethods@section}
\providecommand*{\toclevel@matmethods@section}{0}
% Format is set for research articles by default
\titleformat{\matmethods@section}
   {\sffamily\normalsize\bfseries}
   {}
   {0em}
   {#1}
   []
\titlespacing{\matmethods@section}
	{0pt}
	{3.25ex plus 1ex minus .2ex}
	{1.5ex plus .2ex}
\newcommand{\showmatmethods}{% Display materials&methods section
\@ifundefined{@matmethods}{}{\section*{Materials and Methods}{\sffamily\small\noindent\@matmethods}}
}

%% Other packages
\RequirePackage{enumitem} % For reducing bullet list item separation

%% For sidecaptions
\RequirePackage[rightcaption]{sidecap}

%% Define widetext as a double-column float, with a warning
\RequirePackage{float}
\RequirePackage{stfloats}
\RequirePackage{marginnote}
\floatstyle{plain}
\newfloat{@widetext}{hbt!}{wtt}
\newenvironment{widetext}{%
  \PackageWarning{pnas-new}{Use of `widetext` is not recommended. We will now place it at the top or bottom of a page.}
  \begin{@widetext*}[bt!]
  \marginnote{\itshape\footnotesize\color{red}Use of \texttt{widetext} is not recommended.}
  \hrule
}{
  \hrule
  \end{@widetext*}
}

%% For backward compatibility; does nothing
\def\pnasbreak{}

\endinput
