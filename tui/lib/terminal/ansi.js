const ANSI_REGEX = /\x1B\[[0-?]*[ -/]*[@-~]/g

const strippedAnsi = (str) => (typeof str === 'string' ? str.replace(ANSI_REGEX, '') : '')

const sanitizedLine = (line, maxCols = 100) => {
  const clean = strippedAnsi(line).replace(/\t/g, '    ')
  return clean.length > maxCols ? clean.slice(0, maxCols) : clean
}

module.exports = {
  strippedAnsi,
  sanitizedLine,
}
