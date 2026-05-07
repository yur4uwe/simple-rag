package main

import (
	"crypto/md5"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

const baseURL = "https://htmx.org"

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <page_to_scrape>")
	}

	page := os.Args[1]
	filename := fmt.Sprintf("data/htmx_%s.md", page)

	var err error
	switch page {
	case "docs":
		err = convertPage(baseURL+"/docs/", filename, nil, true)
	case "examples":
		err = convertPage(baseURL+"/examples/", filename, []string{"examples", "server-examples"}, false)
	case "reference":
		err = convertPage(baseURL+"/reference/", filename, []string{"attributes", "headers", "events", "api"}, false)
	case "essays":
		err = convertPage(baseURL+"/essays/", filename, []string{"essays"}, false)
	default:
		log.Fatalf("Unknown page: %s", page)
	}

	if err != nil {
		log.Fatal(err)
	}
}

func convertPage(targetURL, filename string, subPageTypes []string, useColLimit bool) error {
	doc := getPageMain(targetURL)
	if doc == nil {
		return fmt.Errorf("failed to get page: %s", targetURL)
	}

	if useColLimit {
		doc = doc.Find("[class*='10 col']")
		if doc.Length() != 1 {
			return fmt.Errorf("expected exactly one '10 col' element, got %d", doc.Length())
		}
	}

	var sb strings.Builder
	parseBodylikeInner(&sb, doc, "")

	if err := os.WriteFile(filename, []byte(sb.String()), 0644); err != nil {
		return err
	}

	if len(subPageTypes) > 0 {
		fetchReferences(filterReferences(doc, subPageTypes))
	}
	return nil
}

func get_main(r io.Reader) *goquery.Selection {
	doc, err := goquery.NewDocumentFromReader(r)
	if err != nil {
		log.Fatal(err)
	}

	main := doc.Find("main")
	if main.Length() == 0 {
		log.Fatal("Expected at least one 'main' element")
	}
	return main.First()
}

func getPageMain(targetURL string) *goquery.Selection {
	if targetURL == "" {
		return nil
	}

	cacheDir := "data/cache"
	hasher := md5.New()
	hasher.Write([]byte(targetURL))
	filename := filepath.Join(cacheDir, fmt.Sprintf("%x.html", hasher.Sum(nil)))

	if f, err := os.Open(filename); err == nil {
		log.Printf("Cache hit for %s", targetURL)
		defer f.Close()
		return get_main(f)
	}

	log.Printf("Cache miss for %s", targetURL)
	time.Sleep(2 * time.Second)
	res, err := http.Get(targetURL)
	if err != nil || res.StatusCode != 200 {
		log.Fatalf("Request failed: %s (status %v, err %v)", targetURL, res.StatusCode, err)
	}
	defer res.Body.Close()

	os.MkdirAll(cacheDir, 0755)
	f, _ := os.Create(filename)
	defer f.Close()

	return get_main(io.TeeReader(res.Body, f))
}

func parseInfoTable(sb *strings.Builder, table *goquery.Selection) {
	if goquery.NodeName(table) != "table" {
		return
	}

	header := table.Find("th")
	if header.Length() > 0 {
		header.Each(func(i int, s *goquery.Selection) {
			fmt.Fprintf(sb, "| **%s** ", s.Text())
		})
		sb.WriteString("|\n")
		for range header.Length() {
			sb.WriteString("| --- ")
		}
		sb.WriteString("|\n")
	}

	table.Find("tr").Each(func(i int, s *goquery.Selection) {
		if s.Find("td").Length() == 0 {
			return
		}
		s.Children().Each(func(i int, s *goquery.Selection) {
			fmt.Fprintf(sb, "| %s ", s.Text())
		})
		sb.WriteString("|\n")
	})
}

func getCleanText(s *goquery.Selection) string {
	clone := s.Clone()
	clone.Find("a.zola-anchor").Remove()
	return strings.TrimSpace(clone.Text())
}

func parseInline(sb *strings.Builder, p *goquery.Selection) {
	p.Contents().Each(func(i int, node *goquery.Selection) {
		name := goquery.NodeName(node)
		text := node.Text()

		switch name {
		case "#text":
			trimmed := strings.TrimSpace(text)
			if trimmed == "" {
				if sb.Len() > 0 && !strings.HasSuffix(sb.String(), " ") && !strings.HasSuffix(sb.String(), "\n") {
					sb.WriteString(" ")
				}
				return
			}
			if (text[0] == ' ' || text[0] == '\n') && sb.Len() > 0 && !strings.HasSuffix(sb.String(), " ") && !strings.HasSuffix(sb.String(), "\n") {
				sb.WriteString(" ")
			}
			sb.WriteString(trimmed)
			if text[len(text)-1] == ' ' || text[len(text)-1] == '\n' {
				sb.WriteString(" ")
			}
		case "em", "i":
			fmt.Fprintf(sb, "*%s*", strings.TrimSpace(text))
		case "strong", "b":
			fmt.Fprintf(sb, "**%s**", strings.TrimSpace(text))
		case "code":
			fmt.Fprintf(sb, "`%s`", text)
		case "a":
			href, _ := node.Attr("href")
			fmt.Fprintf(sb, "[%s](%s)", strings.TrimSpace(text), href)
		case "br":
			sb.WriteString("\n")
		case "p", "div", "span":
			parseInline(sb, node)
		case "time":
			sb.WriteString(strings.TrimSpace(text))
		default:
			sb.WriteString(text)
		}
	})
}

func parseBodylikeInner(sb *strings.Builder, bodylike *goquery.Selection, prefix string) {
	bodylike.Children().Each(func(i int, s *goquery.Selection) {
		name := goquery.NodeName(s)
		switch name {
		case "p":
			sb.WriteString(prefix)
			parseInline(sb, s)
		case "h1", "h2", "h3", "h4":
			level := strings.Repeat("#", int(name[1]-'0'))
			fmt.Fprintf(sb, "%s%s %s", prefix, level, getCleanText(s))
		case "pre":
			lang, _ := s.Attr("data-lang")
			fmt.Fprintf(sb, "%s```%s\n%s\n%s```", prefix, lang, s.Text(), prefix)
		case "blockquote":
			parseBodylikeInner(sb, s, prefix+"> ")
			return
		case "ul", "ol":
			s.Find("li").Each(func(i int, li *goquery.Selection) {
				fmt.Fprintf(sb, "%s- ", prefix)
				parseInline(sb, li)
				sb.WriteString("\n")
			})
		case "table":
			parseInfoTable(sb, s)
		case "div":
			if s.AttrOr("class", "") == "info-table" {
				parseInfoTable(sb, s.Children())
			}
		}
		sb.WriteString("\n\n")
	})
}

func filterReferences(doc *goquery.Selection, pageTypes []string) *goquery.Selection {
	return doc.Find("a").FilterFunction(func(i int, s *goquery.Selection) bool {
		href, _ := s.Attr("href")
		if strings.HasPrefix(href, "/") {
			href = baseURL + href
			s.SetAttr("href", href)
		}
		path := strings.TrimPrefix(href, baseURL+"/")
		parts := strings.Split(path, "/")
		return len(parts) > 1 && slices.Contains(pageTypes, parts[0])
	})
}

func fetchReferences(references *goquery.Selection) {
	visited := map[string]bool{}
	references.Each(func(i int, s *goquery.Selection) {
		href := s.AttrOr("href", "")
		u, _ := url.Parse(href)
		path := strings.Trim(u.Path, "/")

		if visited[path] || path == "" {
			return
		}
		visited[path] = true

		log.Printf("Scraping reference: %s", href)
		doc := getPageMain(href)
		if doc == nil {
			return
		}

		var sb strings.Builder
		parseBodylikeInner(&sb, doc, "")

		fullPath := filepath.Join("data", path)
		os.MkdirAll(filepath.Dir(fullPath), 0755)
		os.WriteFile(fullPath+".md", []byte(sb.String()), 0644)
	})
}
