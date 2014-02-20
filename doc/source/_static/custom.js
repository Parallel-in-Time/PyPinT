$(document).ready(function () {
  $('#pypint-version.section > h1:not(.jumbotron > h1)').remove();

  $('.jumbotron + .row p:nth-child(1)').addClass('lead');
  $('h1 + p').addClass('lead');

  var $sidebar = $('#sidebar');
  $('#sidebar > ul').each(function() {
    $sidebar.append('<div class="list-group"></div>');
    var $last_group = $('#sidebar > div.list-group').last();

    $(this).find('> li').each(function() {
      $(this).find('> a').each(function() {
        var $target_class = '';
        if ( $(this).attr('href') === '#' ) {
          $target_class = ' active';
        }
        $last_group.append($(this).clone().addClass('list-group-item').addClass($target_class));
      });
      $(this).find('> ul > li > a').each(function() {
        $last_group.append($(this).clone().addClass('list-group-item'));
      });
    });
    $(this).remove();
  });
});
