"""
Crawler for subscene.com

Seed list: subtitle download page for all items with jp subs

1. for each seed:
2.   remove id at end of url (remaineder = jp subs for that item home), visit item home
3.   if singleton choose best sub page & download, else download each episode etc
4.   replace /japanese with /english in url suffix
5.   download equivilant/best subs (if applicable)
"""
