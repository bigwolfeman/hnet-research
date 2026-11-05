// goviews: extract (function code, doc comment) view pairs from a NUL-separated stream of Go files.
//
// Pseudocode:
//
//	for each file (parallel workers, output in input order):
//	    parse with comments; skip files that do not parse
//	    for each FuncDecl with a doc comment:
//	        doc  = comment text (no //), leading function name removed, whitespace collapsed
//	        code = source bytes from the `func` keyword to the end of the body (the doc comment is not included)
//	        keep if minDoc <= len(doc) <= maxDoc and minCode <= len(code) <= maxCode
//	write code.bin and doc.bin (records separated by NUL, same order) and stats to stdout
//
// usage: go run scripts/goviews/main.go -in go_train.bin -out <dir>
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

type pair struct{ code, doc []byte }

func extract(src []byte, minDoc, maxDoc, minCode, maxCode int) []pair {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil
	}
	var out []pair
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Doc == nil || fn.Body == nil {
			continue
		}
		doc := strings.Join(strings.Fields(fn.Doc.Text()), " ")
		if rest, found := strings.CutPrefix(doc, fn.Name.Name+" "); found {
			doc = rest
		}
		start, end := fset.Position(fn.Type.Func).Offset, fset.Position(fn.End()).Offset
		if start < 0 || end > len(src) || start >= end {
			continue
		}
		code := src[start:end]
		if len(doc) < minDoc || len(doc) > maxDoc || len(code) < minCode || len(code) > maxCode ||
			bytes.IndexByte(code, 0) >= 0 || strings.IndexByte(doc, 0) >= 0 {
			continue
		}
		out = append(out, pair{append([]byte(nil), code...), []byte(doc)})
	}
	return out
}

func main() {
	in := flag.String("in", "", "NUL-separated Go files")
	outDir := flag.String("out", "", "output dir")
	minDoc := flag.Int("min-doc", 40, "")
	maxDoc := flag.Int("max-doc", 512, "")
	minCode := flag.Int("min-code", 200, "")
	maxCode := flag.Int("max-code", 2048, "")
	flag.Parse()
	data, err := os.ReadFile(*in)
	if err != nil {
		panic(err)
	}
	files := bytes.Split(data, []byte{0})
	results := make([][]pair, len(files))
	var wg sync.WaitGroup
	jobs := make(chan int, 1024)
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = extract(files[i], *minDoc, *maxDoc, *minCode, *maxCode)
			}
		}()
	}
	for i := range files {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		panic(err)
	}
	cf, _ := os.Create(filepath.Join(*outDir, "code.bin"))
	df, _ := os.Create(filepath.Join(*outDir, "doc.bin"))
	cw, dw := bufio.NewWriterSize(cf, 1<<22), bufio.NewWriterSize(df, 1<<22)
	n, filesWithPairs, codeBytes, docBytes := 0, 0, 0, 0
	for _, ps := range results {
		if len(ps) > 0 {
			filesWithPairs++
		}
		for _, p := range ps {
			cw.Write(p.code)
			cw.WriteByte(0)
			dw.Write(p.doc)
			dw.WriteByte(0)
			n, codeBytes, docBytes = n+1, codeBytes+len(p.code), docBytes+len(p.doc)
		}
	}
	cw.Flush()
	dw.Flush()
	cf.Close()
	df.Close()
	fmt.Printf("{\"files\": %d, \"files_with_pairs\": %d, \"pairs\": %d, \"code_bytes\": %d, \"doc_bytes\": %d}\n",
		len(files), filesWithPairs, n, codeBytes, docBytes)
}
