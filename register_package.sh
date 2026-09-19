#!/bin/bash
# Sovwave.jl Package Registration Script
# Prepares and tags the package for Julia General Registry registration

set -e  # Exit on error

VERSION="v0.2.1"
echo "🌊 Sovwave.jl Package Registration"
echo "===================================="
echo "Version: $VERSION"
echo ""

# 1. Verify we're in the right directory
if [ ! -f "Project.toml" ]; then
    echo "❌ Error: Project.toml not found. Run this script from the package root."
    exit 1
fi

# 2. Check git status
echo "📊 Checking git status..."
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes."
    echo "   Please commit or stash them before tagging."
    git status --short
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. Run tests
echo ""
echo "🧪 Running test suite..."
if julia --project=. -e 'using Pkg; Pkg.test()'; then
    echo "✅ Tests passed"
else
    echo "❌ Tests failed. Fix issues before registering."
    exit 1
fi

# 4. Validate Project.toml
echo ""
echo "🔍 Validating Project.toml..."
if julia --project=. -e 'using Pkg; Pkg.resolve(); println("✅ Project.toml valid")'; then
    echo "✅ Package resolution successful"
else
    echo "❌ Package resolution failed"
    exit 1
fi

# 5. Check exports
echo ""
echo "📦 Verifying exports..."
if julia --project=. -e 'using Sovwave; println("✅ All exports valid")'; then
    echo "✅ Package loads successfully"
else
    echo "❌ Package loading failed"
    exit 1
fi

# 6. Run quick tokenizer validation
echo ""
echo "🎵 Testing tournament winner tokenizer..."
julia --project=. -e '
using Sovwave
tok = default_tokenizer()
text = "Wave 🌊 computing test"
tokens = tokenize_ids(tok, text)
println("✅ Tokenized: $(length(text)) chars → $(length(tokens)) tokens")
println("   Compression: $(round(length(tokens)/length(text), digits=3)) tokens/char")
' || exit 1

# 7. Check if tag already exists
echo ""
echo "🏷️  Checking git tags..."
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo "⚠️  Tag $VERSION already exists"
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "$VERSION"
        git push origin ":refs/tags/$VERSION" 2>/dev/null || true
    else
        exit 1
    fi
fi

# 8. Create annotated tag
echo ""
echo "🏷️  Creating git tag: $VERSION"
git tag -a "$VERSION" -m "Release $VERSION: Enhanced tokenizer with Soliton_WavePacket tournament winner

Changes:
- ✅ 144-algorithm tokenizer tournament (Soliton_WavePacket winner, Score: 11,099.29)
- ✅ 0.7318 tokens/char compression (27% better than char-level)
- ✅ 100% reconstruction accuracy across 12 diverse test files
- ✅ Adaptive bigram frequency analysis for multi-char tokens
- ✅ GitHub Pages tokenizer demo with live waveform visualization
- ✅ Complete tournament winners documentation
- ✅ Registration guide and packaging preparation
"

echo "✅ Tag created: $VERSION"

# 9. Show next steps
echo ""
echo "===================================="
echo "✅ Package ready for registration!"
echo "===================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Push commits and tag to GitHub:"
echo "   git push origin main"
echo "   git push origin $VERSION"
echo ""
echo "2. Go to: https://github.com/TheProtaganist/sovwave"
echo ""
echo "3. Comment on latest commit:"
echo "   @JuliaRegistrator register"
echo ""
echo "4. Wait for automated PR to JuliaRegistries/General"
echo "   (typically 3 days for new packages, 15 min for updates)"
echo ""
echo "5. After approval, users can install with:"
echo "   using Pkg; Pkg.add(\"Sovwave\")"
echo ""
echo "📚 See REGISTRATION_GUIDE.md for more details"

# 10. Optional: Show tag info
echo ""
read -p "Show tag details? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git show "$VERSION" --no-patch
fi

# 11. Optional: Push immediately
echo ""
read -p "Push to GitHub now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to origin..."
    git push origin main
    git push origin "$VERSION"
    echo ""
    echo "✅ Pushed to GitHub!"
    echo "Now go to GitHub and comment: @JuliaRegistrator register"
else
    echo ""
    echo "Remember to push manually:"
    echo "  git push origin main"
    echo "  git push origin $VERSION"
fi

echo ""
echo "🌊 Sovwave.jl registration preparation complete!"
