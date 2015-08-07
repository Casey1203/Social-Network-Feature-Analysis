% Encode in UTF-8

\NeedsTeXFormat{LaTeX2e}[2015/01/01]
\ProvidesClass{hkureport}
[2015/07/15 v0.1 LaTeX template for HKU MSc project report in department of computer science]
\typeout{Document class 'hkureport' v0.1 by Lai{2015/07}}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax

\LoadClass[a4paper,12pt]{book}[2014/09/29]

\RequirePackage{listings}

% Import the global macro 
\RequirePackage{ifthen}
\RequirePackage{url}
% import package for supporting Chinese
\RequirePackage{CJK}
% import package for supporting emoji
\RequirePackage{coloremoji}
% Set the font style
\RequirePackage{palatino}
\RequirePackage{bm}
% use three line table
\RequirePackage{booktabs}
\RequirePackage{multirow}
% Set up margin
\RequirePackage{geometry}
\newgeometry{%
  top=25mm, bottom=25mm, left=38mm, right=38mm
}
\savegeometry{reportgeometry}
\loadgeometry{reportgeometry}

%%%%%%%%Line spacing, indent%%%%%%%%
\RequirePackage{indentfirst}
\renewcommand{\baselinestretch}{1.66} 
\setlength{\parindent}{2em}
\setlength{\abovecaptionskip}{10pt plus 1pt minus 1pt}
\setlength{\belowcaptionskip}{3pt plus 1pt minus 2pt}

\renewcommand{\bibname}{References}

%%%%%%%%header and footer%%%%%%%%
\RequirePackage{fancyhdr}
\fancypagestyle{frontmatter}{%
	\pagestyle{empty}
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{mainmatter}{
  \fancyhead{}
  \fancyfoot{}
  \if@twoside
    \fancyfoot[RO]{\zihao{-5}\thepage}
    \fancyfoot[LE]{\zihao{-5}\thepage} \fancyfoot
  \else
    \fancyfoot[C]{\thepage}
  \fi
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}


%%%%%%%%table of contents%%%%%%%%
\RequirePackage{titletoc}
\titlecontents{chapter}[0pt]{\normalsize}{\thecontentslabel\ }{}
     {\hspace{.5em}\titlerule*[4pt]{.}\contentspage}
\titlecontents{section}[2em]{\vspace{0.1\baselineskip}\normalsize}
{\thecontentslabel\ }{}
     {\hspace{.5em}\titlerule*[4pt]{.}\contentspage}
\titlecontents{subsection}[4em]{\vspace{0.1\baselineskip}\normalsize}
{\thecontentslabel\ }{}
     {\hspace{.5em}\titlerule*[4pt]{.}\contentspage}

%%%%%%%%graphics%%%%%%%%
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\DeclareGraphicsExtensions{.eps,.ps,.jpg,.bmp,.png}
\graphicspath{{figure/}}


%%%%%%%%other settings%%%%%%%%
\RequirePackage{setspace}

% front matter before main body
\renewcommand{\frontmatter}{%
  \clearpage
  \@mainmatterfalse
  \pagestyle{frontmatter}
}

\newcommand{\contentsmatter}{%
  \@mainmattertrue
  \pagestyle{mainmatter}
  \setcounter{page}{1}
  \pagenumbering{Roman}
}

%control column width in table
\RequirePackage{array}
\newcolumntype{I}{!{\vrule width 3pt}}


\renewcommand{\mainmatter}{%
  	\clearpage
	\setcounter{page}{1}
	\pagenumbering{arabic}
}

% Redifine the title page
\newcommand{\titlepg}{
\begin{titlepage}
\begin{center}
\includegraphics[width=2in]{hkulogo.jpg}
\vspace{24pt}

\fontsize{12pt}{\baselineskip}{The University of Hong Kong}
\vspace{12pt}

\fontsize{12pt}{\baselineskip}{Faculty of Engineering}
\vspace{12pt}

\fontsize{12pt}{\baselineskip}{Department of Computer Science}
\vspace{48pt}

\fontsize{12pt}{\baselineskip}{COMP7704}
\vspace{12pt}

\begin{singlespace}
\fontsize{12pt}{\baselineskip}{Dissertation Title}

\fontsize{12pt}{\baselineskip}{\hku@reporttitle}
\end{singlespace}
\vspace{48pt}

\begin{singlespace}
\fontsize{12pt}{\baselineskip}{Submitted in partial fulfillment of the requirements for the admission to\\ the degree of Master of Science in Computer Science}
\end{singlespace}
\vspace{60pt}

\begin{singlespace}
\fontsize{12pt}{\baselineskip}{By}

\fontsize{12pt}{\baselineskip}{Student name: {\hku@author}}

\fontsize{12pt}{\baselineskip}{HKU Student No.: \hku@studentID}
\end{singlespace}
\vspace{12pt}

\begin{singlespace}
\fontsize{12pt}{\baselineskip}{Supervisor: \hku@title\ \hku@supervisorname}

\fontsize{12pt}{\baselineskip}{Date of submission: 01/08/2015}
\end{singlespace}
\end{center}

\include{data/abstract}
\declaration
\include{data/acknowledgements}
\end{titlepage}
}

\renewcommand{\maketitle}{%
	\frontmatter
	\titlepg
}

% Abstract
\newenvironment{abstract}{%
\clearpage
%\thispagestyle{empty}
\vspace*{\baselineskip}
\begin{center}
\Large\textbf{Abstract}
\end{center}
}

% Declaration
\newcommand{\declaration}{%
\clearpage
%\thispagestyle{empty}
\vspace*{\baselineskip}
\begin{center}
\Large\textbf{Declaration}
\end{center}
I declare that this thesis ``\hku@reporttitle'' represents my own work, except where due acknowledgement is made, and that it has not been previously included in a thesis, dissertation or report submitted to this university or to any other institution for a degree, diploma or other qualification.
}

% Acknowledgement
\newenvironment{acknowledgements}{%
\clearpage
%\thispagestyle{empty}
\vspace*{\baselineskip}
\begin{center}
\Large\textbf{Acknowledgements}
\end{center}
}

%%%%%%%%report info%%%%%%%%
% Title
\def\hku@reporttitle{}
\newcommand{\reporttitle}[1]{%
    \def\hku@reporttitle{#1}
}

% student name
\def\hku@author{}
\newcommand{\reportauthor}[1]{%
  \def\hku@author{#1}
}

% student ID no.
\def\hku@studentID{}
\newcommand{\studentID}[1]{%
  \def\hku@studentID{#1}
}

% supervisor
\def\hku@supervisorname{}
\newcommand{\supervisor}[1]{%
  \def\hku@supervisorname{#1}
}

% supervisor title
\def\hku@title{}
\newcommand{\supervisorTitle}[1]{%
  \def\hku@title{#1}
}